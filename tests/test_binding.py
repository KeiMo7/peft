# Copyright 2026-present the HuggingFace Inc. team.
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
"""Tests for opt-in adapter <-> base model binding verification."""

import json
import os

import pytest
import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM

from peft import (
    BindingError,
    LoraConfig,
    MergeVerificationError,
    PeftModel,
    PeftWarning,
    get_peft_model,
)
from peft.utils import binding as binding_utils


# A tiny LLM is used so that the merge check can build `input_ids`; it runs in well under a second on CPU.
def make_config(num_layers=2):
    return LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=num_layers,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )


def make_base(seed=0, num_layers=2):
    torch.manual_seed(seed)
    return LlamaForCausalLM(make_config(num_layers))


def make_lora(**kwargs):
    return LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], init_lora_weights=False, **kwargs)


def save_adapter(tmp_path, name="adapter", save_binding=True, base_seed=0, **config_kwargs):
    model = get_peft_model(make_base(base_seed), make_lora(**config_kwargs))
    path = tmp_path / name
    model.save_pretrained(path, save_binding=save_binding)
    return path


class TestBindingRecord:
    def test_save_without_binding_keeps_weight_format(self, tmp_path):
        # The adapter weight format must not change: the binding record is an extra file, never a change to the weights.
        plain = save_adapter(tmp_path, "plain", save_binding=False)
        bound = save_adapter(tmp_path, "bound", save_binding=True)

        assert sorted(os.listdir(plain)) == ["README.md", "adapter_config.json", "adapter_model.safetensors"]
        assert sorted(os.listdir(bound)) == sorted(sorted(os.listdir(plain)) + ["adapter_binding.json"])

        from safetensors.torch import load_file

        plain_weights = load_file(plain / "adapter_model.safetensors")
        bound_weights = load_file(bound / "adapter_model.safetensors")
        assert plain_weights.keys() == bound_weights.keys()
        for key in plain_weights:
            assert torch.equal(plain_weights[key], bound_weights[key])

    def test_record_contents(self, tmp_path):
        path = save_adapter(tmp_path)
        with open(path / "adapter_binding.json", encoding="utf-8") as f:
            record = json.load(f)

        assert record["format_version"] == 1
        assert record["base_model"]["weight_digest"]
        assert record["target_modules"] == [
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.v_proj",
            "model.layers.1.self_attn.q_proj",
            "model.layers.1.self_attn.v_proj",
        ]
        assert record["adapter"]["num_tensors"] > 0
        assert record["adapter"]["digest"]
        assert record["peft_type"] == "LORA"

    def test_digest_is_deterministic_and_content_based(self):
        state = {name: tensor.clone() for name, tensor in make_base().state_dict().items()}
        first = binding_utils.digest_state_dict(state)["digest"]
        second = binding_utils.digest_state_dict(state)["digest"]
        assert first == second

        # Insertion order must not matter.
        reordered = dict(reversed(list(state.items())))
        assert binding_utils.digest_state_dict(reordered)["digest"] == first

        # Content changes must matter.
        changed = dict(state)
        changed["model.embed_tokens.weight"] = changed["model.embed_tokens.weight"] + 1
        assert binding_utils.digest_state_dict(changed)["digest"] != first

    def test_digest_cache_is_invalidated_by_in_place_changes(self):
        model = make_base()
        state = dict(model.named_parameters())
        cache = {}
        first = binding_utils.digest_state_dict(state, cache=cache)["digest"]
        # An unchanged model reuses the cached digests.
        assert binding_utils.digest_state_dict(state, cache=cache)["digest"] == first
        assert cache

        with torch.no_grad():
            model.model.embed_tokens.weight.add_(1)
        assert binding_utils.digest_state_dict(dict(model.named_parameters()), cache=cache)["digest"] != first

    def test_base_digest_is_cached_on_the_model(self):
        model = get_peft_model(make_base(), make_lora())
        binding_utils.current_binding(model)
        cache = getattr(model.get_base_model(), binding_utils._DIGEST_CACHE_ATTR)
        # A second call must reuse the memoized digest instead of re-hashing the base weights.
        assert binding_utils.current_binding(model)["base_model"]["weight_digest"] == cache["digest"]["digest"]


class TestBindingLoad:
    def test_matching_base_loads(self, tmp_path):
        path = save_adapter(tmp_path)
        model = PeftModel.from_pretrained(make_base(), path, verify_binding=True)
        listing = model.list_adapter_bindings()
        assert listing["default"]["status"] == "verified"
        assert listing["default"]["differences"] == []

    def test_different_base_revision_is_rejected(self, tmp_path):
        # Two models with an identical config but different weights stand in for two revisions of the same base model.
        path = save_adapter(tmp_path, base_seed=0)
        with pytest.raises(BindingError, match="different base model"):
            PeftModel.from_pretrained(make_base(seed=1), path, verify_binding=True)

    def test_different_base_revision_allowed_with_force_load(self, tmp_path):
        path = save_adapter(tmp_path, base_seed=0)
        with pytest.warns(PeftWarning, match="force_load"):
            model = PeftModel.from_pretrained(make_base(seed=1), path, verify_binding=True, force_load=True)
        assert model.list_adapter_bindings()["default"]["status"] == "mismatch"

    def test_explicit_revision_mismatch_is_rejected(self, tmp_path):
        # The recorded revision is part of the identity even when the weights happen to be identical. The revision of a
        # base model that came from a Hub snapshot is available as `config._commit_hash`.
        path = save_adapter(tmp_path)
        with open(path / "adapter_binding.json", encoding="utf-8") as f:
            record = json.load(f)
        record["base_model"]["revision"] = "v2"
        with open(path / "adapter_binding.json", "w", encoding="utf-8") as f:
            json.dump(record, f)

        base = make_base()
        base.config._commit_hash = "v1"
        with pytest.raises(BindingError, match="revision"):
            PeftModel.from_pretrained(base, path, verify_binding=True)

    def test_unknown_current_revision_is_not_a_mismatch(self, tmp_path):
        # A base model that was not loaded from a snapshot has no revision. The weight digest is then the only identity
        # check; an unknown current revision must not reject an otherwise matching adapter.
        path = save_adapter(tmp_path)
        with open(path / "adapter_binding.json", encoding="utf-8") as f:
            record = json.load(f)
        record["base_model"]["revision"] = "v2"
        with open(path / "adapter_binding.json", "w", encoding="utf-8") as f:
            json.dump(record, f)

        base = make_base()
        assert getattr(base.config, "_commit_hash", None) is None
        model = PeftModel.from_pretrained(base, path, verify_binding=True)
        assert model.list_adapter_bindings()["default"]["status"] == "verified"

    def test_legacy_adapter_without_metadata_warns_and_can_be_forced(self, tmp_path):
        path = save_adapter(tmp_path, save_binding=False)
        with pytest.raises(BindingError, match="no adapter_binding.json"):
            PeftModel.from_pretrained(make_base(), path, verify_binding=True)

        with pytest.warns(PeftWarning, match="binding to this base model is unknown"):
            model = PeftModel.from_pretrained(make_base(), path, verify_binding=True, force_load=True)
        assert model.list_adapter_bindings()["default"]["status"] == "unknown"

    def test_target_module_missing_from_base_is_an_error(self, tmp_path):
        # A record that names a module the base model does not have must not be silently ignored.
        path = save_adapter(tmp_path)
        with open(path / "adapter_binding.json", encoding="utf-8") as f:
            record = json.load(f)
        record["target_modules"].append("model.layers.0.self_attn.nonexistent_proj")
        with open(path / "adapter_binding.json", "w", encoding="utf-8") as f:
            json.dump(record, f)

        with pytest.raises(BindingError, match="do not exist in the base model"):
            PeftModel.from_pretrained(make_base(), path, verify_binding=True)

    def test_verification_off_keeps_existing_behavior(self, tmp_path):
        # Without `verify_binding=True` an adapter on a different base loads exactly as before.
        path = save_adapter(tmp_path, base_seed=0)
        model = PeftModel.from_pretrained(make_base(seed=1), path)
        assert "default" in model.peft_config
        assert not hasattr(model, "_adapter_bindings")

    def test_saved_inputs_output_are_unchanged(self, tmp_path):
        # The verification path must not alter the model output.
        path = save_adapter(tmp_path)
        inputs = torch.randint(0, 64, (2, 4))
        plain = PeftModel.from_pretrained(make_base(), path).eval()
        verified = PeftModel.from_pretrained(make_base(), path, verify_binding=True).eval()
        with torch.no_grad():
            assert torch.equal(plain(inputs).logits, verified(inputs).logits)
            with verified.disable_adapter():
                disabled = verified(inputs).logits
            base_only = make_base().eval()
            assert torch.allclose(disabled, base_only(inputs).logits, atol=1e-6, rtol=1e-6)

    def test_load_adapter_verifies_too(self, tmp_path):
        path = save_adapter(tmp_path)
        model = PeftModel.from_pretrained(make_base(), path, adapter_name="first", verify_binding=True)
        model.load_adapter(path, "second", verify_binding=True)
        listing = model.list_adapter_bindings()
        assert set(listing) == {"first", "second"}
        assert all(entry["status"] == "verified" for entry in listing.values())

        # An adapter trained on a different base is rejected through `load_adapter` as well.
        other = save_adapter(tmp_path, "other", base_seed=3)
        with pytest.raises(BindingError):
            model.load_adapter(other, "third", verify_binding=True)


class TestBindingListingAndConflicts:
    def test_listing_reports_each_adapter(self, tmp_path):
        first = save_adapter(tmp_path, "first")
        second = save_adapter(tmp_path, "second")
        model = PeftModel.from_pretrained(make_base(), first, adapter_name="alpha", verify_binding=True)
        model.load_adapter(second, "beta", verify_binding=True)

        listing = model.list_adapter_bindings()
        assert set(listing) == {"alpha", "beta"}
        assert listing["alpha"]["base_model"]["weight_digest"] == listing["beta"]["base_model"]["weight_digest"]
        assert listing["alpha"]["active"] is True
        assert listing["beta"]["active"] is False
        assert listing["alpha"]["source"] == os.fspath(first)

    def test_conflicting_adapters_name_both_parties(self, tmp_path):
        first = save_adapter(tmp_path, "first", base_seed=0)
        second = save_adapter(tmp_path, "second", base_seed=1)
        model = PeftModel.from_pretrained(make_base(), first, adapter_name="alpha", verify_binding=True)

        # The second adapter targets the same modules but is bound to different base weights: report both parties.
        record = binding_utils.read_binding(second)
        with pytest.raises(BindingError) as excinfo:
            binding_utils.check_conflicts(model, "beta", record, force_load=False)
        message = str(excinfo.value)
        assert "'alpha'" in message and "'beta'" in message
        assert "q_proj" in message

    def test_conflict_is_raised_by_load_adapter(self, tmp_path):
        # End to end: loading an adapter that was trained on different base weights onto a model that already has a
        # verified adapter for the same modules is rejected with both adapter names in the message.
        first = save_adapter(tmp_path, "first", base_seed=0)
        second = save_adapter(tmp_path, "second", base_seed=1)
        model = PeftModel.from_pretrained(make_base(), first, adapter_name="alpha", verify_binding=True)
        # Make the second record compatible with the live base so that only the cross-adapter conflict remains.
        record = binding_utils.read_binding(second)
        record["base_model"] = binding_utils.current_binding(model)["base_model"]
        with open(second / "adapter_binding.json", "w", encoding="utf-8") as f:
            json.dump(record, f)
        # Give the first adapter a different recorded base so the two disagree (without freezing the real digest).
        stored = dict(model._adapter_bindings["alpha"])
        stored["base_model"] = dict(stored["base_model"], weight_digest="0" * 64)
        model._adapter_bindings["alpha"] = stored

        with pytest.raises(BindingError) as excinfo:
            model.load_adapter(second, "beta", verify_binding=True)
        message = str(excinfo.value)
        assert "alpha" in message and "beta" in message


class TestMergeVerification:
    def test_verified_merge_matches_unmerged_output(self, tmp_path):
        path = save_adapter(tmp_path)
        model = PeftModel.from_pretrained(make_base(), path, verify_binding=True).eval()
        inputs = torch.randint(0, 64, (2, 4))
        with torch.no_grad():
            unmerged = model(inputs).logits

        model.merge_adapter(verify_merge=True)
        with torch.no_grad():
            merged = model(inputs).logits
        assert torch.allclose(unmerged, merged, atol=1e-4, rtol=1e-4)

    def test_verified_merge_and_unload(self, tmp_path):
        path = save_adapter(tmp_path)
        model = PeftModel.from_pretrained(make_base(), path, verify_binding=True).eval()
        inputs = torch.randint(0, 64, (2, 4))
        with torch.no_grad():
            expected = model(inputs).logits
        merged = model.merge_and_unload(verify_merge=True)
        with torch.no_grad():
            assert torch.allclose(expected, merged(inputs).logits, atol=1e-4, rtol=1e-4)

    def test_failed_verification_rolls_back(self, tmp_path, monkeypatch):
        path = save_adapter(tmp_path)
        model = PeftModel.from_pretrained(make_base(), path, verify_binding=True).eval()
        before = {name: tensor.clone() for name, tensor in model.get_base_model().state_dict().items()}

        # Force the comparison to fail without touching the merge machinery.
        monkeypatch.setattr(binding_utils, "_outputs_close", lambda first, second: (False, "forced mismatch"))

        with pytest.raises(MergeVerificationError, match="rolled back"):
            model.merge_adapter(verify_merge=True)

        after = model.get_base_model().state_dict()
        assert before.keys() == after.keys()
        for name in before:
            assert torch.equal(before[name], after[name]), name
        assert not any(getattr(module, "merged", False) for module in model.modules())
        # The model still produces the unmerged adapter output.
        inputs = torch.randint(0, 64, (2, 4))
        with torch.no_grad():
            assert torch.isfinite(model(inputs).logits).all()

    def test_unverified_merge_path_is_unchanged(self, tmp_path):
        path = save_adapter(tmp_path)
        model = PeftModel.from_pretrained(make_base(), path).eval()
        model.merge_adapter()
        assert any(getattr(module, "merged", False) for module in model.modules())


class TestBindingOnCustomModels:
    """The binding check must not require a transformers model."""

    class MLP(nn.Module):
        def __init__(self, seed=0):
            super().__init__()
            torch.manual_seed(seed)
            self.lin0 = nn.Linear(16, 32)
            self.lin1 = nn.Linear(32, 4)

        def forward(self, x):
            return self.lin1(torch.relu(self.lin0(x)))

    def test_custom_model(self, tmp_path):
        config = LoraConfig(r=4, target_modules=["lin0", "lin1"], init_lora_weights=False)
        torch.manual_seed(0)
        model = get_peft_model(self.MLP(seed=0), config)
        path = tmp_path / "adapter"
        model.save_pretrained(path, save_binding=True)
        record = binding_utils.read_binding(path)
        assert record["target_modules"] == ["lin0", "lin1"]

        loaded = PeftModel.from_pretrained(self.MLP(seed=0), path, verify_binding=True)
        inputs = torch.randn(2, 16)
        with torch.no_grad():
            assert torch.isfinite(loaded(inputs)).all()

        with pytest.raises(BindingError):
            PeftModel.from_pretrained(self.MLP(seed=5), path, verify_binding=True)

    def test_custom_model_merge_verification_falls_back_to_weights(self, tmp_path):
        config = LoraConfig(r=4, target_modules=["lin0", "lin1"], init_lora_weights=False)
        model = get_peft_model(self.MLP(seed=0), config)
        path = tmp_path / "adapter"
        model.save_pretrained(path)
        reloaded = PeftModel.from_pretrained(self.MLP(seed=0), path).eval()
        inputs = torch.randn(2, 16)
        with torch.no_grad():
            expected = reloaded(inputs)
        reloaded.merge_adapter(verify_merge=True)
        with torch.no_grad():
            assert torch.allclose(expected, reloaded(inputs), atol=1e-4, rtol=1e-4)


class TestMergeVerificationRollback:
    def test_merge_call_that_fails_half_way_is_rolled_back(self, tmp_path, monkeypatch):
        # A merge that raises after touching some layers must not leave a partially merged model behind.
        path = save_adapter(tmp_path)
        model = PeftModel.from_pretrained(make_base(), path).eval()
        before = {name: tensor.clone() for name, tensor in model.get_base_model().state_dict().items()}

        original = model.base_model.merge_adapter
        calls = {"n": 0}

        def failing_merge(*args, **kwargs):
            original(*args, **kwargs)
            calls["n"] += 1
            raise RuntimeError("boom after merging")

        monkeypatch.setattr(model.base_model, "merge_adapter", failing_merge)
        with pytest.raises(RuntimeError, match="boom"):
            model.merge_adapter(verify_merge=True)

        after = model.get_base_model().state_dict()
        for name in before:
            assert torch.equal(before[name], after[name]), name
        assert not any(getattr(module, "merged", False) for module in model.modules())

    def test_merge_invalidates_the_memoized_digest(self, tmp_path):
        # Merging changes the base weights in place with `.data`, which does not bump the tensor version counters the
        # digest cache relies on. A stale cache would report the merged model as still matching the binding.
        path = save_adapter(tmp_path)
        model = PeftModel.from_pretrained(make_base(), path, verify_binding=True)
        assert model.list_adapter_bindings()["default"]["status"] == "verified"
        model.merge_adapter()
        listing = model.list_adapter_bindings()["default"]
        assert listing["status"] == "mismatch"
        assert any("weight digest" in difference for difference in listing["differences"])

    def test_merge_verification_falls_back_for_encoder_decoder(self, tmp_path):
        # Encoder-decoder models need more than `input_ids`; the check falls back to comparing weights, exactly.
        from transformers import T5Config, T5ForConditionalGeneration

        config = T5Config(
            vocab_size=64,
            d_model=16,
            d_ff=32,
            num_layers=1,
            num_decoder_layers=1,
            num_heads=2,
            decoder_start_token_id=0,
        )
        torch.manual_seed(0)
        base = T5ForConditionalGeneration(config)
        lora_config = LoraConfig(r=4, target_modules=["q", "v"], init_lora_weights=False)
        model = get_peft_model(base, lora_config)
        path = tmp_path / "t5_adapter"
        model.save_pretrained(path, save_binding=True)

        torch.manual_seed(0)
        reloaded = PeftModel.from_pretrained(T5ForConditionalGeneration(config), path, verify_binding=True).eval()
        inputs = {"input_ids": torch.randint(0, 64, (1, 4)), "decoder_input_ids": torch.randint(0, 64, (1, 3))}
        with torch.no_grad():
            expected = reloaded(**inputs).logits
        reloaded.merge_adapter(verify_merge=True)
        with torch.no_grad():
            assert torch.allclose(expected, reloaded(**inputs).logits, atol=1e-3, rtol=1e-3)

    def test_binding_record_does_not_replace_the_weights(self, tmp_path):
        # An adapter directory whose weights were removed still has its binding record, but cannot be loaded: the
        # record supplements the weights, it does not stand in for them.
        path = save_adapter(tmp_path)
        os.remove(path / "adapter_model.safetensors")
        with pytest.raises((ValueError, OSError)):
            PeftModel.from_pretrained(make_base(), os.fspath(path), verify_binding=True)
