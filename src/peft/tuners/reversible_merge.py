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

"""Reversible (snapshot-based) adapter merge.

The regular :meth:`BaseTuner.merge_adapter` / :meth:`BaseTuner.unmerge_adapter` pair mutates the base weights
in-place (``weight += delta`` / ``weight -= delta``). In reduced precision (float16 / bfloat16) each add/subtract
incurs a fresh rounding, so the unmerged weight drifts away from the original. The drift accumulates linearly with
the number of merge/unmerge cycles, which makes a canary rollback ("unmerge after merge") numerically unsafe.

The reversible merge implemented here stores an exact snapshot of every affected base parameter before the first
merge. Unmerge restores that snapshot bit-for-bit instead of subtracting a recomputed delta, therefore:

* the round-trip error of the *base weights* is exactly zero, independent of the number of cycles;
* half-precision merge arithmetic can be upcast to float32 to reduce the error of the *merged* weights;
* a checksum recorded at merge time detects external modifications of the merged weights before unmerge.

The bookkeeping lives entirely outside the inference hot path: while a layer is merged, its forward pass is the
plain base-layer forward, exactly as with the regular merge.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
from typing import Any, Optional

import torch
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file
from torch import nn

from .tuners_utils import BaseTunerLayer, onload_layer


# Sidecar files written next to the adapter weights. They only exist when a reversible merge snapshot is active,
# so checkpoints of models that never use reversible_merge remain byte-identical to before.
SIDECAR_TENSORS_FILE = "reversible_merge_state.safetensors"
SIDECAR_META_FILE = "reversible_merge_state.json"


class ReversibleMergeError(RuntimeError):
    """Raised when a reversible merge cannot be performed or cannot be reversed safely.

    The error message always names the layer and the concrete reason, and whenever the error is raised mid-merge the
    model is rolled back to its pre-merge state first.
    """


@dataclasses.dataclass
class LayerMergeState:
    """Bookkeeping for one tuner layer under reversible merge."""

    layer_name: str
    merged_adapters: list[str]
    # Exact, bit-for-bit copies of the base parameters taken before the first merge, keyed by parameter name
    # ("weight" and, where applicable, "bias").
    snapshots: dict[str, torch.Tensor]
    # Fingerprints of the merged parameters, used to detect external modification at unmerge time.
    checksums: dict[str, dict[str, Any]]
    cycles: int = 0

    def to_meta(self) -> dict[str, Any]:
        return {
            "layer_name": self.layer_name,
            "merged_adapters": list(self.merged_adapters),
            "checksums": self.checksums,
            "cycles": self.cycles,
            "snapshot_keys": list(self.snapshots.keys()),
        }

    @classmethod
    def from_meta(cls, meta: dict[str, Any], snapshots: dict[str, torch.Tensor]) -> LayerMergeState:
        return cls(
            layer_name=meta["layer_name"],
            merged_adapters=list(meta["merged_adapters"]),
            snapshots=snapshots,
            checksums=dict(meta["checksums"]),
            cycles=int(meta["cycles"]),
        )


class ReversibleMergeRegistry:
    """Holds the reversible-merge bookkeeping of one :class:`BaseTuner`."""

    def __init__(self) -> None:
        # layer name -> LayerMergeState
        self._states: dict[str, LayerMergeState] = {}
        self.upcast_merge: bool = True

    def __bool__(self) -> bool:
        # True while at least one layer is currently merged. After an unmerge the historical states (cycle counts)
        # are retained but no layer is merged anymore.
        return bool(self.merged_layer_names())

    def __contains__(self, layer_name: str) -> bool:
        return layer_name in self._states

    @property
    def states(self) -> dict[str, LayerMergeState]:
        return self._states

    def merged_layer_names(self) -> list[str]:
        return [name for name, state in self._states.items() if state.merged_adapters]

    def get(self, layer_name: str) -> Optional[LayerMergeState]:
        return self._states.get(layer_name)

    def clear(self) -> None:
        self._states.clear()

    def status(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of the current reversible-merge state."""
        layers = [
            {
                "layer_name": state.layer_name,
                "merged_adapters": list(state.merged_adapters),
                "round_trips": state.cycles,
            }
            for state in self._states.values()
        ]
        return {
            "merged": bool(self.merged_layer_names()),
            "upcast_merge": self.upcast_merge,
            "total_layers": len(layers),
            "layers": layers,
        }

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        tensors: dict[str, torch.Tensor] = {}
        meta: dict[str, Any] = {"upcast_merge": self.upcast_merge, "layers": []}
        # Only layers that are currently merged carry snapshots needed for an unmerge elsewhere.
        for state in self._states.values():
            if not state.merged_adapters:
                continue
            for param_name, tensor in state.snapshots.items():
                tensors[f"{state.layer_name}.{param_name}"] = tensor.contiguous().cpu()
            meta["layers"].append(state.to_meta())
        safe_save_file(tensors, os.path.join(directory, SIDECAR_TENSORS_FILE))
        with open(os.path.join(directory, SIDECAR_META_FILE), "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, directory: str, device: Optional[torch.device] = None) -> ReversibleMergeRegistry:
        meta_path = os.path.join(directory, SIDECAR_META_FILE)
        if not os.path.exists(meta_path):
            raise FileNotFoundError(meta_path)
        with open(meta_path) as f:
            meta = json.load(f)
        tensor_path = os.path.join(directory, SIDECAR_TENSORS_FILE)
        tensors = safe_load_file(tensor_path, device=str(device) if device is not None else "cpu")
        registry = cls()
        registry.upcast_merge = bool(meta.get("upcast_merge", True))
        for layer_meta in meta["layers"]:
            layer_name = layer_meta["layer_name"]
            snapshots = {
                param_name: tensors[f"{layer_name}.{param_name}"] for param_name in layer_meta["snapshot_keys"]
            }
            registry._states[layer_name] = LayerMergeState.from_meta(layer_meta, snapshots)
        return registry

    @staticmethod
    def sidecar_exists(directory: str) -> bool:
        return os.path.exists(os.path.join(directory, SIDECAR_META_FILE))


def _checksum(tensor: torch.Tensor) -> dict[str, Any]:
    """Cheap, deterministic fingerprint of a tensor (not a cryptographic hash).

    Two independent moments in float64 plus shape and dtype make accidental collisions from ordinary weight
    modifications effectively impossible, while keeping the cost to one reduction per parameter.
    """
    t = tensor.detach()
    if t.is_floating_point() or t.is_complex():
        t64 = t.double()
        return {
            "dtype": str(t.dtype),
            "shape": list(t.shape),
            "sum": float(t64.sum().item()),
            "sumsq": float((t64 * t64).sum().item()),
            "numel": int(t.numel()),
        }
    t64 = t.long()
    return {
        "dtype": str(t.dtype),
        "shape": list(t.shape),
        "sum": int(t64.sum().item()),
        "sumsq": int((t64 * t64).sum().item()),
        "numel": int(t.numel()),
    }


def _checksums_match(expected: dict[str, Any], tensor: torch.Tensor, *, rtol: float = 1e-6) -> bool:
    actual = _checksum(tensor)
    if actual["dtype"] != expected["dtype"] or actual["shape"] != expected["shape"]:
        return False
    if actual["numel"] != expected["numel"]:
        return False
    for key in ("sum", "sumsq"):
        exp, got = expected[key], actual[key]
        if not math.isfinite(exp) or not math.isfinite(got):
            if exp != got:
                return False
        elif abs(got - exp) > rtol * max(1.0, abs(exp)):
            return False
    return True


def _base_parameters(layer: BaseTunerLayer) -> dict[str, nn.Parameter]:
    base_layer = layer.get_base_layer()
    params = {"weight": base_layer.weight}
    if getattr(base_layer, "bias", None) is not None:
        params["bias"] = base_layer.bias
    return params


def _is_grouped_conv(layer: BaseTunerLayer) -> bool:
    base_layer = layer.get_base_layer()
    return getattr(base_layer, "groups", 1) > 1


def _is_quantized_base(layer: BaseTunerLayer) -> bool:
    base_layer = layer.get_base_layer()
    # Quantized base layers store their weights outside of a plain `.weight` parameter (qweight, qpacked, ...) or
    # expose quantization backend markers; merging/reversing them cannot rely on plain tensor arithmetic.
    if hasattr(base_layer, "qweight") or hasattr(base_layer, "qweight_packed") or hasattr(base_layer, "W_q"):
        return True
    if getattr(base_layer, "quantization_backend", None) is not None:
        return True
    weight = getattr(base_layer, "weight", None)
    return weight is not None and getattr(weight, "is_quantized", False)


def _require_mergeable(layer_name: str, layer: BaseTunerLayer) -> None:
    """Pre-flight check: raise before any weight is touched if this layer cannot be merged reversibly."""
    if _is_quantized_base(layer):
        raise ReversibleMergeError(
            f"Reversible merge is not supported for quantized layer '{layer_name}' "
            f"({type(layer.get_base_layer()).__name__}); its weights are not stored as plain tensors."
        )
    if _is_grouped_conv(layer):
        raise ReversibleMergeError(
            f"Reversible merge is not supported for grouped convolution layer '{layer_name}' "
            f"(groups={layer.get_base_layer().groups})."
        )


def _upcast_merge_into(layer: BaseTunerLayer, adapter_names: list[str], upcast: bool) -> None:
    """Merge the adapters into the base weights, optionally performing the arithmetic in float32.

    The base parameters keep their original dtypes: they are temporarily replaced by float32 copies, the layer's own
    ``merge`` runs once on those copies (so every supported PEFT method keeps its merge formula), and the results are
    cast back. In float16/bfloat16 this avoids rounding the ``weight + delta`` addition in low precision.
    """
    base_layer = layer.get_base_layer()
    params = _base_parameters(layer)
    need_upcast = upcast and any(p.data.dtype in (torch.float16, torch.bfloat16) for p in params.values())
    originals = {name: p.data for name, p in params.items()}
    if need_upcast:
        for name, param in params.items():
            param.data = originals[name].float()
    try:
        layer.merge(adapter_names=adapter_names)
    finally:
        if need_upcast:
            for name, param in params.items():
                merged = param.data
                param.data = originals[name]
                param.data.copy_(merged.to(originals[name].dtype))


def _iter_tuner_layers(tuner) -> list[tuple[str, BaseTunerLayer]]:
    layers: list[tuple[str, BaseTunerLayer]] = []
    prefix = "model."  # BaseTuner wraps the user model under `.model`
    for name, module in tuner.named_modules():
        if isinstance(module, BaseTunerLayer):
            short_name = name.removeprefix(prefix)
            layers.append((short_name, module))
    return layers


def reversible_merge(
    tuner,
    adapter_names: Optional[list[str]] = None,
    *,
    upcast_merge: bool = True,
    registry: ReversibleMergeRegistry,
) -> None:
    """Merge adapters while keeping exact pre-merge snapshots, with all-or-nothing semantics."""
    layers = _iter_tuner_layers(tuner)
    if not layers:
        return

    # Resolve the adapter selection once; mirror check_adapters_to_merge semantics but reject mixing with a plain
    # (non-reversible) merge, whose bookkeeping the registry cannot reason about.
    selected = adapter_names
    newly_merged: list[tuple[str, BaseTunerLayer, list[str], dict[str, torch.Tensor]]] = []
    preflight_errors: list[str] = []

    for layer_name, layer in layers:
        with onload_layer(layer):
            active = list(layer.active_adapters) if selected is None else list(selected)
            if layer.merged:
                if registry.get(layer_name) is None:
                    preflight_errors.append(
                        f"layer '{layer_name}' is already merged through the non-reversible merge API; "
                        "unmerge it before using reversible merge"
                    )
                continue
            if not active:
                continue
            try:
                _require_mergeable(layer_name, layer)
            except ReversibleMergeError as exc:
                preflight_errors.append(str(exc))
    if preflight_errors:
        raise ReversibleMergeError(
            "Reversible merge rejected before any weight was modified:\n- " + "\n- ".join(preflight_errors)
        )

    registry.upcast_merge = upcast_merge
    for layer_name, layer in layers:
        with onload_layer(layer):
            active = list(layer.active_adapters) if selected is None else list(selected)
            if layer.merged:
                # A layer already merged under this registry stays merged (e.g. merging a second adapter on top);
                # a layer merged through the legacy API was rejected in the pre-flight pass above.
                if registry.get(layer_name) is not None:
                    continue
            if not active:
                continue
            params = _base_parameters(layer)
            snapshots = {name: param.detach().clone() for name, param in params.items()}
            try:
                _upcast_merge_into(layer, active, upcast_merge)
                # Non-finite values must abort and roll back, even though the regular unsafe merge does not check.
                for name, param in _base_parameters(layer).items():
                    if not torch.isfinite(param.data).all():
                        raise ReversibleMergeError(
                            f"Non-finite values appeared in '{layer_name}.{name}' while merging adapters {active}."
                        )
            except Exception:
                # Roll back every layer touched in this call, including ones that had already been merged.
                for prev_name, prev_layer, _, prev_snapshots in newly_merged:
                    with onload_layer(prev_layer):
                        for pname, snapshot in prev_snapshots.items():
                            _base_parameters(prev_layer)[pname].data.copy_(snapshot)
                        prev_layer.merged_adapters.clear()
                        registry._states.pop(prev_name, None)
                for name, param in params.items():
                    param.data.copy_(snapshots[name])
                layer.merged_adapters.clear()
                raise
            checksums = {name: _checksum(param.data) for name, param in _base_parameters(layer).items()}
            registry._states[layer_name] = LayerMergeState(
                layer_name=layer_name,
                merged_adapters=list(layer.merged_adapters),
                snapshots=snapshots,
                checksums=checksums,
                cycles=registry.get(layer_name).cycles if registry.get(layer_name) else 0,
            )
            newly_merged.append((layer_name, layer, active, snapshots))


def reversible_unmerge(tuner, *, registry: ReversibleMergeRegistry) -> None:
    """Restore the exact pre-merge snapshots, refusing if the merged weights were modified externally."""
    if not registry:
        return
    layers = dict(_iter_tuner_layers(tuner))
    problems: list[str] = []
    for layer_name, state in registry._states.items():
        layer = layers.get(layer_name)
        if layer is None:
            problems.append(f"layer '{layer_name}' recorded in the merge state no longer exists in the model")
            continue
        with onload_layer(layer):
            for param_name, expected in state.checksums.items():
                param = _base_parameters(layer)[param_name]
                if not _checksums_match(expected, param.data):
                    problems.append(
                        f"merged weights of '{layer_name}.{param_name}' do not match the state recorded at merge "
                        "time; they appear to have been modified externally, so unmerge is refused"
                    )
    if problems:
        raise ReversibleMergeError(
            "Reversible unmerge refused; the model is left unchanged:\n- " + "\n- ".join(problems)
        )

    for layer_name, state in registry._states.items():
        layer = layers[layer_name]
        with onload_layer(layer):
            for param_name, snapshot in state.snapshots.items():
                _base_parameters(layer)[param_name].data.copy_(snapshot.to(layer.get_base_layer().weight.device))
            layer.merged_adapters.clear()
            state.merged_adapters = []
            state.checksums = {}
            state.cycles += 1
