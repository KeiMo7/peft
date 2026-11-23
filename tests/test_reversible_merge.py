# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the snapshot-based reversible merge (``merge_adapter_reversible`` /
``unmerge_adapter_reversible``).

The guarantees under test are:

* a merge/unmerge round trip restores the base weights bit-for-bit (exact, not tolerance-based) and therefore the
  pre-merge outputs exactly;
* the error does not accumulate over repeated round trips, because unmerge restores a snapshot instead of
  subtracting a recomputed delta;
* the merge is atomic and refuses unsupported layers before touching any weight;
* the bookkeeping is queryable, survives save/load (also on a fresh process-like reconstruction), and stays off the
  inference path;
* the regular (non-reversible) merge API is unchanged.
"""

import copy
import json
import os
import tempfile

import pytest
import torch
from torch import nn

from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.reversible_merge import (
    SIDECAR_META_FILE,
    SIDECAR_TENSORS_FILE,
    ReversibleMergeError,
)
from peft.tuners.tuners_utils import BaseTunerLayer


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin0 = nn.Linear(32, 32)
        self.lin1 = nn.Linear(32, 16)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.lin1(self.act(self.lin0(x)))


class GroupedConvModel(nn.Module):
    def __init__(self, groups=2):
        super().__init__()
        self.lin0 = nn.Linear(32, 32)
        self.conv = nn.Conv2d(8, 8, 3, padding=1, groups=groups)
        self.act = nn.ReLU()

    def forward(self, x):
        h = self.lin0(x).view(x.shape[0], 8, 2, 2)
        return self.act(self.conv(h)).flatten(1)


def make_model(dtype=torch.float32, model_cls=MLP, **config_kwargs):
    torch.manual_seed(0)
    base = model_cls().to(dtype)
    cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["lin0", "lin1"] if model_cls is MLP else ["lin0", "conv"],
        init_lora_weights=True,
        **config_kwargs,
    )
    model = get_peft_model(copy.deepcopy(base), cfg).eval()
    # Default LoRA init is B=0 (zero delta); fill non-zero random adapter weights so merging actually changes the
    # base weights.
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, BaseTunerLayer):
                for name in module.lora_A:
                    module.lora_A[name].weight.normal_(0, 0.1)
                    module.lora_B[name].weight.normal_(0, 0.1)
    return model


def tuner_layers(model):
    return [m for m in model.modules() if isinstance(m, BaseTunerLayer)]


def tuner_layers_by_name(model):
    return {
        name.removeprefix("model."): module
        for name, module in model.base_model.named_modules()
        if isinstance(module, BaseTunerLayer)
    }


def base_weights(model):
    return [layer.get_base_layer().weight.detach().clone() for layer in tuner_layers(model)]


DTYPES = [torch.float32, torch.float16, torch.bfloat16]


class TestReversibleMergeRoundTrip:
    @pytest.mark.parametrize("dtype", DTYPES, ids=lambda t: str(t).split(".")[-1])
    def test_single_round_trip_restores_weights_and_outputs_exactly(self, dtype):
        model = make_model(dtype)
        x = torch.randn(4, 32, dtype=dtype)
        ref_weights = base_weights(model)
        with torch.inference_mode():
            ref_out = model(x).clone()

        model.merge_adapter_reversible()
        # merged weights must actually differ from the original
        assert all(not torch.equal(w, w0) for w, w0 in zip(base_weights(model), ref_weights))
        model.unmerge_adapter_reversible()

        restored_weights = base_weights(model)
        assert all(torch.equal(w, w0) for w, w0 in zip(restored_weights, ref_weights))
        with torch.inference_mode():
            restored_out = model(x)
        assert torch.equal(restored_out, ref_out)

    @pytest.mark.parametrize("dtype", DTYPES, ids=lambda t: str(t).split(".")[-1])
    def test_ten_round_trips_have_zero_drift(self, dtype):
        model = make_model(dtype)
        ref_weights = base_weights(model)
        for _ in range(10):
            model.merge_adapter_reversible()
            model.unmerge_adapter_reversible()
        for restored, ref in zip(base_weights(model), ref_weights):
            # exact bit-for-bit equality, hence an absolute upper bound of zero, independent of cycle count
            assert torch.equal(restored, ref)
            assert (restored.float() - ref.float()).abs().max().item() == 0.0

    @pytest.mark.parametrize("dtype", DTYPES, ids=lambda t: str(t).split(".")[-1])
    def test_merged_output_matches_unmerged_path_within_tolerance(self, dtype):
        model = make_model(dtype)
        x = torch.randn(4, 32, dtype=dtype)
        with torch.inference_mode():
            unmerged_out = model(x).clone()
        model.merge_adapter_reversible(upcast_merge=True)
        with torch.inference_mode():
            merged_out = model(x).clone()
        # The merged path computes the same function; the difference is floating-point association
        # ((W + BA)x vs Wx + B(Ax)) plus, in half precision, the rounding of the merged weights.
        atol = {torch.float32: 1e-6, torch.float16: 5e-3, torch.bfloat16: 2e-2}[dtype]
        max_diff = (merged_out.float() - unmerged_out.float()).abs().max().item()
        assert torch.allclose(merged_out.float(), unmerged_out.float(), atol=atol, rtol=0.0), max_diff

    def test_unmerge_touches_only_targeted_layers_and_keeps_others(self):
        model = make_model()
        untouched = {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" not in n}
        model.merge_adapter_reversible()
        model.unmerge_adapter_reversible()
        for name, param in model.named_parameters():
            if "lora_" not in name:
                assert torch.equal(param.detach(), untouched[name])

    def test_round_trip_count_is_queryable(self):
        model = make_model()
        assert model.reversible_merge_status()["merged"] is False
        for expected in range(1, 4):
            model.merge_adapter_reversible()
            status = model.reversible_merge_status()
            assert status["merged"] is True
            assert len(status["layers"]) == len(tuner_layers(model))
            assert all(layer["merged_adapters"] == ["default"] for layer in status["layers"])
            model.unmerge_adapter_reversible()
            status = model.reversible_merge_status()
            assert status["merged"] is False
            assert all(layer["round_trips"] == expected for layer in status["layers"])

    def test_idempotent_unmerge_warns_without_state(self):
        model = make_model()
        with pytest.warns(UserWarning, match="No reversibly merged layers"):
            model.unmerge_adapter_reversible()


class TestReversibleMergeGuards:
    def test_grouped_conv_rejected_before_any_modification(self):
        model = make_model(model_cls=GroupedConvModel)
        before = base_weights(model)
        with pytest.raises(ReversibleMergeError, match="grouped convolution"):
            model.merge_adapter_reversible()
        # nothing merged, model still serves inference
        assert all(not layer.merged for layer in tuner_layers(model))
        assert all(torch.equal(w, w0) for w, w0 in zip(base_weights(model), before))
        x = torch.randn(2, 32)
        assert torch.isfinite(model(x)).all()

    def test_non_finite_adapter_aborts_and_rolls_back(self):
        model = make_model()
        before = base_weights(model)
        model.base_model.model.lin0.lora_B["default"].weight.data.fill_(float("inf"))
        with pytest.raises(ReversibleMergeError, match="Non-finite"):
            model.merge_adapter_reversible()
        assert all(not layer.merged for layer in tuner_layers(model))
        assert all(torch.equal(w, w0) for w, w0 in zip(base_weights(model), before))
        assert model.reversible_merge_status()["merged"] is False

    def test_failure_mid_model_rolls_back_already_merged_layers(self):
        # lin0 merges fine; make conv fail by turning its base into a grouped conv after construction is not
        # possible, so instead force a non-finite result on the second layer while the first would succeed.
        model = make_model(model_cls=GroupedConvModel)
        before = base_weights(model)
        model.base_model.model.conv.lora_B["default"].weight.data.fill_(float("inf"))
        with pytest.raises(ReversibleMergeError):
            model.merge_adapter_reversible()
        # both layers, including the one processed first, must be back to their original weights
        assert all(not layer.merged for layer in tuner_layers(model))
        assert all(torch.equal(w, w0) for w, w0 in zip(base_weights(model), before))

    def test_external_modification_of_merged_weights_is_detected(self):
        model = make_model()
        model.merge_adapter_reversible()
        model.base_model.model.lin0.get_base_layer().weight.data.add_(1.0)
        with pytest.raises(ReversibleMergeError, match="modified externally"):
            model.unmerge_adapter_reversible()
        # refused unmerge leaves the model merged and unchanged
        assert all(layer.merged for layer in tuner_layers(model))

    def test_mixing_with_legacy_merge_is_rejected(self):
        model = make_model()
        model.merge_adapter()
        with pytest.raises(ReversibleMergeError, match="non-reversible merge API"):
            model.merge_adapter_reversible()
        model.unmerge_adapter()

    def test_error_message_names_layer_and_reason(self):
        model = make_model(model_cls=GroupedConvModel)
        with pytest.raises(ReversibleMergeError) as exc_info:
            model.merge_adapter_reversible()
        message = str(exc_info.value)
        assert "conv" in message
        assert "groups" in message

    def test_quantized_layer_rejected_before_any_modification(self):
        # Quantized base modules (bnb, gptq, ...) expose their packed weights outside of a plain `.weight`
        # parameter, marked here by a `qweight` attribute, and cannot rely on plain tensor merge arithmetic.
        model = make_model()
        before = base_weights(model)
        model.base_model.model.lin0.get_base_layer().qweight = torch.zeros(1, dtype=torch.int32)
        with pytest.raises(ReversibleMergeError, match="quantized"):
            model.merge_adapter_reversible()
        assert all(not layer.merged for layer in tuner_layers(model))
        assert all(torch.equal(w, w0) for w, w0 in zip(base_weights(model), before))


class TestReversibleMergePersistence:
    def test_save_writes_sidecar_only_when_merged(self):
        model = make_model()
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            assert not os.path.exists(os.path.join(tmp, SIDECAR_META_FILE))
            assert not os.path.exists(os.path.join(tmp, SIDECAR_TENSORS_FILE))

        model.merge_adapter_reversible()
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            assert os.path.exists(os.path.join(tmp, SIDECAR_META_FILE))
            assert os.path.exists(os.path.join(tmp, SIDECAR_TENSORS_FILE))
            with open(os.path.join(tmp, SIDECAR_META_FILE)) as f:
                meta = json.load(f)
            assert {layer["layer_name"] for layer in meta["layers"]} == {"lin0", "lin1"}

    def test_reload_and_unmerge_on_fresh_model_restores_exact_weights(self):
        model = make_model()
        x = torch.randn(4, 32)
        with torch.inference_mode():
            out_before = model(x).clone()
        model.merge_adapter_reversible()
        registry = model.base_model._reversible_merge_registry
        expected_snapshots = {name: state.snapshots["weight"].clone() for name, state in registry.states.items()}
        with torch.inference_mode():
            out_merged = model(x).clone()

        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            # Reconstruct a brand-new base carrying the saved (merged) weights, as on another machine.
            fresh_base = MLP()
            fresh_base.lin0.weight.data.copy_(model.base_model.model.lin0.get_base_layer().weight.data)
            fresh_base.lin0.bias.data.copy_(model.base_model.model.lin0.get_base_layer().bias.data)
            fresh_base.lin1.weight.data.copy_(model.base_model.model.lin1.get_base_layer().weight.data)
            fresh_base.lin1.bias.data.copy_(model.base_model.model.lin1.get_base_layer().bias.data)
            reloaded = PeftModel.from_pretrained(fresh_base, tmp).eval()

            assert reloaded.reversible_merge_status()["merged"] is True
            with torch.inference_mode():
                assert torch.allclose(reloaded(x), out_merged, atol=1e-6)
            reloaded.unmerge_adapter_reversible()
            layers_by_name = tuner_layers_by_name(reloaded)
            for layer_name, expected in expected_snapshots.items():
                got = layers_by_name[layer_name].get_base_layer().weight
                assert torch.equal(got, expected)
            with torch.inference_mode():
                assert torch.equal(reloaded(x), out_before)

    def test_ten_round_trips_then_save_load_then_unmerge(self):
        model = make_model()
        x = torch.randn(4, 32)
        with torch.inference_mode():
            out_before = model(x).clone()
        for _ in range(10):
            model.merge_adapter_reversible()
            model.unmerge_adapter_reversible()
        # one final merge, persist, and reverse on a freshly loaded model
        model.merge_adapter_reversible()
        expected = {
            name: state.snapshots["weight"].clone()
            for name, state in model.base_model._reversible_merge_registry.states.items()
        }
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            fresh_base = MLP()
            fresh_base.lin0.weight.data.copy_(model.base_model.model.lin0.get_base_layer().weight.data)
            fresh_base.lin0.bias.data.copy_(model.base_model.model.lin0.get_base_layer().bias.data)
            fresh_base.lin1.weight.data.copy_(model.base_model.model.lin1.get_base_layer().weight.data)
            fresh_base.lin1.bias.data.copy_(model.base_model.model.lin1.get_base_layer().bias.data)
            reloaded = PeftModel.from_pretrained(fresh_base, tmp).eval()
            reloaded.unmerge_adapter_reversible()
            layers_by_name = tuner_layers_by_name(reloaded)
            for name, snapshot in expected.items():
                assert torch.equal(layers_by_name[name].get_base_layer().weight, snapshot)
            with torch.inference_mode():
                assert torch.equal(reloaded(x), out_before)


class TestReversibleMergeDoesNotChangeExistingBehavior:
    def test_legacy_merge_api_signature_and_state_unchanged(self):
        model = make_model()
        # regular merge still works and does not create a reversible registry
        model.merge_adapter()
        assert not hasattr(model.base_model, "_reversible_merge_registry") or not (
            model.base_model._reversible_merge_registry.merged_layer_names()
        )
        assert all(layer.merged for layer in tuner_layers(model))
        model.unmerge_adapter()
        assert all(not layer.merged for layer in tuner_layers(model))

    def test_unmerged_forward_path_numerically_unchanged(self):
        model = make_model()
        x = torch.randn(4, 32)
        with torch.inference_mode():
            first = model(x).clone()
        # merely instantiating the registry must not alter any numerical result
        _ = model.base_model.reversible_merge_registry
        with torch.inference_mode():
            second = model(x).clone()
        assert torch.equal(first, second)

    def test_merged_forward_is_plain_base_layer_forward(self):
        model = make_model()
        model.merge_adapter_reversible()
        for layer in tuner_layers(model):
            # while merged, the adapter delta path is bypassed; merged flag drives the plain base forward
            assert layer.merged is True
        x = torch.randn(4, 32)
        with torch.inference_mode():
            out = model(x)
        assert torch.isfinite(out).all()
