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
"""Adapter <-> base model binding metadata.

A PEFT checkpoint only contains adapter weights; nothing ties them to the exact base model they were trained on. Loading
an adapter into a different revision (or a different model that happens to have the same name) therefore silently
produces wrong results. This module records and verifies that binding. All of it is opt-in:

- ``save_pretrained(save_binding=True)`` writes a small ``adapter_binding.json`` next to the adapter weights. The adapter
  weight format itself is unchanged.
- ``from_pretrained(verify_binding=True)`` / ``load_adapter(verify_binding=True)`` check the record before the adapter is
  injected. ``force_load=True`` downgrades mismatches to warnings.
- ``merge_adapter(verify_merge=True)`` / ``merge_and_unload(verify_merge=True)`` check that the merged weights equal what
  the unmerged adapter computes, and roll the merge back on failure.

The weight digest is content based: tensor names, dtypes, shapes, and bytes are hashed in a fixed (sorted) order. It is
deterministic across processes and threads, does not use random numbers, and is cached on the base model so that loading
several adapters into the same base model only hashes the base once. Sharded tensors (e.g. FSDP2 ``DTensor``) are not
hashed, because a local shard does not identify the global weights.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import warnings
from typing import Any, Optional

import torch


BINDING_NAME = "adapter_binding.json"
BINDING_FORMAT_VERSION = 1
# Bump when the digest algorithm changes; records with a different version are treated as unverifiable.
DIGEST_ALGORITHM = "sha256/v1"

_EMPTY_DIGEST = hashlib.sha256(b"peft-binding-empty/v1").hexdigest()
# Attribute on the base model that memoizes its digest. It is stored as a non-tensor attribute so it is not part of any
# state dict.
_DIGEST_CACHE_ATTR = "_peft_binding_digest_cache"


def _peft_version() -> str:
    try:
        return importlib.metadata.version("peft")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _is_unhashable(tensor: torch.Tensor) -> bool:
    # DTensor (FSDP2 / TP) only exposes a local shard here, so its bytes do not identify the global weights. Meta
    # tensors have no data at all.
    return tensor.__class__.__name__ == "DTensor" or tensor.device.type == "meta"


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    # `.view(torch.uint8)` is a zero-copy reinterpretation, but `numpy()` needs host memory and does not support every
    # dtype, so go through a CPU copy. One tensor's bytes are materialized at a time, never the whole state dict.
    cpu = tensor.detach().contiguous().reshape(-1).cpu()
    return cpu.view(torch.uint8).numpy().tobytes()


def digest_state_dict(
    state_dict: dict[str, torch.Tensor],
    cache: Optional[dict[str, tuple[str, str]]] = None,
) -> dict[str, Any]:
    """Return a deterministic digest of a state dict.

    Args:
        state_dict (`dict[str, torch.Tensor]`):
            The tensors to digest. Keys must be the canonical (original) module names.
        cache (`dict[str, tuple[str, str]]`, *optional*):
            Per-tensor digest cache. When given, it is used for and updated in place: a tensor is re-digested only when
            its version counter, storage, dtype, shape, or device changed. The version counter is bumped by in-place
            autograd-tracked ops, which is what adapter updates and normal training use.

    Returns:
        `dict`: ``{"digest": str, "num_tensors": int, "num_bytes": int, "unhashable": list[str]}``.
    """
    hasher = hashlib.sha256()
    hasher.update(DIGEST_ALGORITHM.encode())
    num_bytes = 0
    skipped: list[str] = []
    for key in sorted(state_dict):
        tensor = state_dict[key]
        if not isinstance(tensor, torch.Tensor):
            continue
        if _is_unhashable(tensor):
            skipped.append(key)
            continue
        signature = (
            f"{key}|{tensor.dtype!s}|{tuple(tensor.shape)}|{tensor.data_ptr()}|"
            f"{getattr(tensor, '_version', -1)}|{tensor.device.type}"
        )
        cached = None if cache is None else cache.get(key)
        if cached is not None:
            cached_signature, cached_digest = cached
            if cached_signature == signature:
                hasher.update(cached_digest.encode())
                num_bytes += tensor.numel() * tensor.element_size()
                continue
        tensor_digest = hashlib.sha256(_tensor_bytes(tensor)).hexdigest()
        if cache is not None:
            cache[key] = (signature, tensor_digest)
        hasher.update(tensor_digest.encode())
        num_bytes += tensor.numel() * tensor.element_size()

    if not state_dict:
        digest = _EMPTY_DIGEST
    else:
        digest = hasher.hexdigest()
    return {
        "digest": digest,
        "num_tensors": len(state_dict) - len(skipped),
        "num_bytes": num_bytes,
        "unhashable": skipped,
    }


def _base_model_source(model: torch.nn.Module) -> dict[str, Any]:
    """Read the identity information the base model exposes, without loading anything from the Hub."""
    name = None
    revision = None
    config = getattr(model, "config", None)
    if config is not None:
        name = getattr(config, "_name_or_path", None) or None
        # Some architectures store the revision of the base weights in the config when it came from a snapshot.
        revision = getattr(config, "_commit_hash", None) or None
    if name is None:
        name = getattr(model, "name_or_path", None) or None
    if name is not None and os.path.isdir(name):
        # A local directory is not an identifier: it can be overwritten with different weights. The digest covers it.
        name = os.path.basename(os.path.normpath(name)) or name
    return {"name": name, "revision": revision}


def compute_base_binding(
    base_model: torch.nn.Module,
    target_modules: Optional[list[str]] = None,
    source: Optional[dict[str, Any]] = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Build the binding record for the base model an adapter is attached to.

    Args:
        base_model (`torch.nn.Module`):
            The *unwrapped* base model, i.e. without any tuner layers. The digest must not include adapter weights,
            otherwise it would differ between adapters and would change whenever an adapter is trained.
        target_modules (`list[str]`, *optional*):
            The modules this adapter targets, as actual module names resolved during injection.
        source (`dict`, *optional*):
            Overrides for the recorded ``name`` / ``revision``.
        use_cache (`bool`, *optional*):
            Reuse the digest memoized on the model when its parameter versions are unchanged.

    Returns:
        `dict`: The JSON-serializable binding record.
    """
    base_source = _base_model_source(base_model)

    state_dict = base_state_dict(base_model)
    signature = tuple(
        (name, str(t.dtype), tuple(t.shape), t.data_ptr(), getattr(t, "_version", -1))
        for name, t in state_dict.items()
    )
    cache = getattr(base_model, _DIGEST_CACHE_ATTR, None) if use_cache else None
    if cache is not None and cache.get("signature") == signature:
        digest = cache["digest"]
    else:
        digest = digest_state_dict(state_dict)
        if use_cache:
            try:
                setattr(base_model, _DIGEST_CACHE_ATTR, {"signature": signature, "digest": digest})
            except AttributeError:
                # Some exotic modules forbid setting attributes; the digest is simply recomputed next time.
                pass

    # Sharded (DTensor) or meta tensors cannot be hashed, so a digest would not describe the whole model (and shards
    # differ per rank). Record it as unavailable instead of recording something that is not a property of the model.
    weight_digest = None if digest["unhashable"] else digest["digest"]
    if source is not None:
        # The caller knows where the base came from (the PEFT config records the Hub id and revision). Use it only when
        # the live model cannot tell; the digest remains the authoritative check in all cases.
        for key in ("name", "revision"):
            if source.get(key) is not None:
                base_source[key] = source[key]
    return {
        "format_version": BINDING_FORMAT_VERSION,
        "digest_algorithm": DIGEST_ALGORITHM,
        "base_model": {
            "name": base_source["name"],
            "revision": base_source["revision"],
            "num_parameters": int(sum(t.numel() for t in state_dict.values())),
            "weight_digest": weight_digest,
            "unhashable_tensors": len(digest["unhashable"]),
        },
        "target_modules": None if target_modules is None else sorted(target_modules),
        "peft_version": _peft_version(),
    }


def compute_adapter_summary(adapter_state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Summarize the adapter weights that will be written to disk."""
    digest = digest_state_dict(adapter_state_dict)
    return {
        "digest": digest["digest"],
        "num_tensors": digest["num_tensors"],
        "num_parameters": int(sum(t.numel() for t in adapter_state_dict.values() if isinstance(t, torch.Tensor))),
    }


def binding_path(directory: str | os.PathLike) -> str:
    return os.path.join(os.fspath(directory), BINDING_NAME)


def save_binding(directory: str | os.PathLike, binding: dict[str, Any]) -> str:
    """Write ``adapter_binding.json`` into ``directory``. Returns the path."""
    path = binding_path(directory)
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(binding, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def read_binding(model_id: str | os.PathLike, **hf_hub_download_kwargs: Any) -> Optional[dict[str, Any]]:
    """Read the binding record for an adapter, locally or from the Hub. Returns ``None`` if it is absent."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError, OfflineModeIsEnabled

    subfolder = hf_hub_download_kwargs.get("subfolder")
    path = os.path.join(os.fspath(model_id), subfolder) if subfolder is not None else os.fspath(model_id)
    local_path = binding_path(path)
    if os.path.isfile(local_path):
        with open(local_path, encoding="utf-8") as f:
            return json.load(f)

    if not os.path.isdir(path) and os.path.isdir(os.fspath(model_id)):
        return None
    if os.path.isdir(os.fspath(model_id)):
        # Local adapter directory without a binding record: an adapter saved by an older PEFT version, or saved without
        # `save_binding=True`. Do not try the Hub.
        return None

    kwargs = dict(hf_hub_download_kwargs)
    kwargs.setdefault("user_agent", "peft")
    kwargs.pop("local_files_only", None)
    try:
        downloaded = hf_hub_download(model_id, BINDING_NAME, **kwargs)
    except OfflineModeIsEnabled:
        # Offline and not cached: the record may exist, but it cannot be read. Do not mistake this for "no record".
        return _unavailable()
    except (EntryNotFoundError, LocalEntryNotFoundError, OSError, ValueError):
        return None
    with open(downloaded, encoding="utf-8") as f:
        return json.load(f)


def _unavailable() -> dict[str, Any]:
    """Marker for a binding record that exists but could not be read (e.g. Hub offline and not cached)."""
    return {"status": "unavailable"}


def _format_difference(label: str, expected: Any, actual: Any) -> Optional[str]:
    if expected == actual:
        return None
    if label == "weight digest" and isinstance(expected, str) and isinstance(actual, str):
        return f"{label}: recorded {expected[:12]}... but current base is {actual[:12]}..."
    return f"{label}: recorded {expected!r} but current base is {actual!r}"


def compare_bindings(recorded: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """Return a list of human-readable differences between a recorded and a current binding."""
    differences: list[str] = []

    recorded_base = recorded.get("base_model") or {}
    current_base = current.get("base_model") or {}
    for key, label in (
        ("name", "base model name"),
        ("revision", "base model revision"),
        ("weight_digest", "weight digest"),
        ("num_parameters", "parameter count"),
    ):
        # A missing value on either side (e.g. an adapter trained from a config-only model, or a base model whose
        # weights are sharded and therefore cannot be digested) is not a contradiction.
        if recorded_base.get(key) is None or current_base.get(key) is None:
            continue
        diff = _format_difference(label, recorded_base.get(key), current_base.get(key))
        if diff:
            differences.append(diff)

    # Only compare the module lists when both sides recorded them. `prepare_binding_check` additionally verifies that
    # every recorded module exists on the base model, which is the rigorous check.
    if recorded.get("target_modules") is not None and current.get("target_modules") is not None:
        recorded_modules = set(recorded["target_modules"])
        current_modules = set(current["target_modules"])
        missing = sorted(recorded_modules - current_modules)
        extra = sorted(current_modules - recorded_modules)
        if missing:
            differences.append(f"target modules missing from the base model: {missing}")
        if extra:
            differences.append(f"target modules not recorded in the adapter: {extra}")

    return differences


def check_adapter_conflicts(bindings: dict[str, dict[str, Any]], target_modules_by_adapter: Optional[dict] = None):
    """Raise [`BindingError`] if two adapters that target the same module are bound to different base models.

    Args:
        bindings (`dict[str, dict]`):
            Mapping from adapter name to binding record.
        target_modules_by_adapter (`dict[str, list[str]]`, *optional*):
            Mapping from adapter name to the modules it actually targets. Defaults to the recorded target modules.
    """
    from .error import BindingError

    modules_by_adapter = target_modules_by_adapter or {
        name: list(recorded.get("target_modules") or []) for name, recorded in bindings.items()
    }
    names = sorted(bindings)
    conflicts = []
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            shared = sorted(set(modules_by_adapter.get(first) or []) & set(modules_by_adapter.get(second) or []))
            if not shared:
                continue
            first_base = bindings[first].get("base_model") or {}
            second_base = bindings[second].get("base_model") or {}
            mismatch = [
                key
                for key in ("name", "revision", "weight_digest")
                if first_base.get(key) is not None
                and second_base.get(key) is not None
                and first_base.get(key) != second_base.get(key)
            ]
            if mismatch:
                details = ", ".join(f"{key}: {first_base.get(key)!r} vs {second_base.get(key)!r}" for key in mismatch)
                conflicts.append(
                    f"adapters {first!r} and {second!r} both target {len(shared)} module(s) "
                    f"(e.g. {shared[0]!r}) but are bound to different base models ({details})"
                )
    if conflicts:
        raise BindingError("Conflicting adapter bindings. " + ". ".join(conflicts) + ".")


def warn(message: str) -> None:
    from .warning import PeftWarning

    warnings.warn(message, PeftWarning, stacklevel=3)


#############################################
# BASE MODEL ACCESS AND TARGET MODULE NAMES #
#############################################


def unwrap_base_model(peft_model: torch.nn.Module) -> torch.nn.Module:
    """Return the base model without the tuner wrapper (``LoraModel`` and friends)."""
    from ..tuners.tuners_utils import BaseTuner

    base = peft_model.base_model
    # Only unwrap known PEFT wrappers. Duck-typing on `.model` would unwrap e.g. a prompt-learning PeftModel, whose
    # `base_model` is the transformers model itself (``LlamaForCausalLM.model`` is the decoder).
    if isinstance(base, BaseTuner) or base.__class__.__name__ == "AdaptionPromptModel":
        return base.model
    return base


def _iter_base_tensors(model: torch.nn.Module):
    """Yield ``(logical_name, tensor)`` for the base model's own parameters and buffers.

    Tuner layers are descended into through their ``base_layer`` and auxiliary wrappers through their
    ``original_module``, and the wrapper infix is dropped from the name. This makes the digest of the base model
    identical before and after adapter injection, and identical for two adapters on the same base model.
    """
    from ..tuners.tuners_utils import BaseTunerLayer
    from .other import AuxiliaryTrainingWrapper

    def walk(module: torch.nn.Module, prefix: str):
        # Tuner layers are ``nn.Module`` subclasses whose ``__init__`` may not call ``nn.Module.__init__``, so their
        # ``named_children`` can be empty. Descend into ``base_layer`` explicitly instead of relying on the module tree.
        if isinstance(module, BaseTunerLayer):
            yield from walk(module.base_layer, prefix)
            return
        if isinstance(module, AuxiliaryTrainingWrapper):
            yield from walk(module.original_module, prefix)
            return
        for name, tensor in list(module.named_parameters(recurse=False)) + list(module.named_buffers(recurse=False)):
            yield f"{prefix}.{name}" if prefix else name, tensor
        for child_name, child in module.named_children():
            yield from walk(child, f"{prefix}.{child_name}" if prefix else child_name)

    yield from walk(model, "")


def base_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return dict(_iter_base_tensors(model))


def _base_module_names(model: torch.nn.Module) -> set[str]:
    """All module names of the base model, with tuner / wrapper infixes stripped from the paths."""
    from ..tuners.tuners_utils import BaseTunerLayer

    names = set()
    for name, module in model.named_modules():
        names.add(name)
        if isinstance(module, BaseTunerLayer):
            base_layer = module.base_layer
            while isinstance(base_layer, BaseTunerLayer):
                base_layer = base_layer.base_layer
            for sub_name, _ in base_layer.named_modules():
                if sub_name:
                    names.add(f"{name}.{sub_name}")
    return names


def target_modules_for_adapter(peft_model: torch.nn.Module, adapter_name: str) -> list[str]:
    """Return the modules that ``adapter_name`` is actually injected into.

    Names are relative to the unwrapped base model (``model.layers.0.self_attn.q_proj``), independent of the
    ``base_model.`` / tuner prefix, so they can be compared across models.
    """
    from ..tuners.tuners_utils import BaseTunerLayer

    modules = []
    for name, module in peft_model.named_modules():
        if not isinstance(module, BaseTunerLayer):
            continue
        try:
            available = module._get_available_adapters()
        except (AttributeError, NotImplementedError):
            available = set()
        if adapter_name in available:
            # Drop the tuner wrapper infixes (``base_layer.``) and the ``base_model.model.`` prefix.
            parts = [part for part in name.split(".") if part != "base_layer"]
            modules.append(".".join(parts).removeprefix("base_model.model."))
    return sorted(modules)


def build_binding(
    peft_model: torch.nn.Module,
    adapter_name: str,
    adapter_state_dict: dict[str, torch.Tensor],
    config: Any,
) -> dict[str, Any]:
    """Build the binding record written next to an adapter."""
    base = unwrap_base_model(peft_model)
    binding = compute_base_binding(
        base,
        target_modules=target_modules_for_adapter(peft_model, adapter_name),
        source={
            "name": getattr(config, "base_model_name_or_path", None),
            "revision": getattr(config, "revision", None),
        },
    )
    binding["adapter"] = compute_adapter_summary(adapter_state_dict)
    binding["peft_type"] = getattr(getattr(config, "peft_type", None), "value", None) or str(
        getattr(config, "peft_type", None)
    )
    binding["config_target_modules"] = _jsonable_target_modules(getattr(config, "target_modules", None))
    return binding


def _jsonable_target_modules(target_modules: Any) -> Any:
    if isinstance(target_modules, (set, tuple)):
        return sorted(target_modules)
    return target_modules


###################################
# MERGE FORWARD-EQUIVALENCE CHECK #
###################################


def _example_inputs(model: torch.nn.Module):
    """Build a small synthetic input for a transformers-like base model.

    Randomness comes from a local generator with a fixed seed, so the global RNG state is never touched and every call
    uses the same inputs.
    """
    config = getattr(model, "config", None)
    if config is None or not hasattr(model, "get_input_embeddings"):
        return None
    try:
        embeddings = model.get_input_embeddings()
    except (AttributeError, NotImplementedError):
        return None
    if not isinstance(embeddings, torch.nn.Embedding):
        return None

    vocab_size = embeddings.weight.shape[0]
    max_position = getattr(config, "max_position_embeddings", None) or getattr(config, "n_positions", None)
    batch, seq_len = 2, 4
    if max_position is not None:
        seq_len = max(1, min(seq_len, int(max_position)))
    if vocab_size < batch * seq_len:
        return None
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    input_ids = torch.randint(0, vocab_size, (batch, seq_len), generator=generator, device="cpu")
    device = next(model.parameters()).device
    inputs = {"input_ids": input_ids.to(device), "attention_mask": torch.ones_like(input_ids)}
    if int(getattr(config, "type_vocab_size", 0) or 0) > 1:
        inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])
    return inputs


def _run_forward(model: torch.nn.Module, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    with torch.no_grad():
        output = model(**inputs)
    # Task heads return `.logits`; feature-extraction heads return `last_hidden_state` / `pooler_output`.
    for attr in ("logits", "last_hidden_state", "pooler_output"):
        value = getattr(output, attr, None)
        if isinstance(value, torch.Tensor):
            return value
    if isinstance(output, (tuple, list)):
        for value in output:
            if isinstance(value, torch.Tensor):
                return value
    if isinstance(output, torch.Tensor):
        return output
    raise TypeError(f"Cannot run a merge verification on an output of type {type(output).__name__}.")


def _outputs_close(first: torch.Tensor, second: torch.Tensor) -> tuple[bool, str]:
    if first.shape != second.shape:
        return False, f"output shape {tuple(first.shape)} vs {tuple(second.shape)}"
    first, second = first.float(), second.float()
    atol = rtol = 1e-4
    if torch.allclose(first, second, atol=atol, rtol=rtol):
        return True, ""
    diff = (first - second).abs()
    outside = (diff > atol + rtol * second.abs()).float().mean().item()
    return False, f"max abs diff {diff.max().item():.3e}, {outside:.2%} of entries outside tolerance"


def _merge_candidates(base_model: torch.nn.Module):
    """Yield ``(name, module)`` for every module whose base weight a merge can change."""
    from ..tuners.tuners_utils import BaseTunerLayer

    for name, module in base_model.named_modules():
        if isinstance(module, BaseTunerLayer) and not name.endswith(".base_layer"):
            yield name, module


def _any_layer_merged(base_model: torch.nn.Module) -> bool:
    from ..tuners.tuners_utils import BaseTunerLayer

    return any(isinstance(module, BaseTunerLayer) and module.merged for module in base_model.modules())


def _base_tensor_snapshot(base_model: torch.nn.Module):
    """Clone the tensors a merge can modify, so a merge/unmerge round trip can be compared exactly."""
    snapshot = []
    for name, module in _merge_candidates(base_model):
        base_layer = module.get_base_layer()
        for attr in ("weight", "bias", "in_proj_weight"):
            value = getattr(base_layer, attr, None)
            if isinstance(value, torch.Tensor):
                snapshot.append((f"{name}.{attr}", value.detach().clone(), base_layer, attr))
    return snapshot


def _verify_by_round_trip(base_model, merge_call, unmerge_call, rollback) -> None:
    """Method-agnostic check for models whose forward cannot take ``input_ids``.

    Merge, unmerge, and require that the unmerge removes exactly the update the merge applied (i.e. restores the
    pre-merge weights). This verifies that the merge wrote to the same weights the matching unmerge reads, without
    assuming anything about how a given PEFT method computes its update (additive, multiplicative, or overwrite).
    Merge and unmerge are deterministic, so the caller keeps the merge by performing it again afterwards.
    """
    from .error import MergeVerificationError

    baseline = _base_tensor_snapshot(base_model)
    merge_call()
    try:
        unmerge_call()
        for name, before, owner, attr in baseline:
            after = getattr(owner, attr)
            # A merge may legitimately be a no-op (e.g. LN tuning swaps in a copy that is still identical before any
            # training), so only the round trip is checked: unmerging must restore exactly the pre-merge weights.
            if not torch.allclose(after, before, atol=1e-5, rtol=1e-5):
                raise MergeVerificationError(f"Merge verification failed: unmerging did not restore {name!r}.")
    except BaseException:
        rollback()
        raise


def verify_merge_inplace(
    base_model: torch.nn.Module,
    merge_call,
    unmerge_call,
    *,
    adapter_names: Optional[list[str]] = None,
    keep_merge: bool = True,
) -> None:
    """Merge, verify that the merged weights reproduce the unmerged output, and roll back on failure.

    Args:
        base_model (`torch.nn.Module`):
            The unwrapped base model, used to run the reference forward pass.
        merge_call (`Callable`):
            Performs the in-place merge.
        unmerge_call (`Callable`):
            Reverts the in-place merge (resetting the tuner bookkeeping).
        adapter_names (`list[str]`, *optional*):
            The adapters that are merged.
        keep_merge (`bool`, *optional*, defaults to `True`):
            Keep the merge when verification passes (``merge_adapter`` semantics). Set to `False` for a trial run
            before an irreversible unload.

    Raises:
        MergeVerificationError:
            If the merge fails or the merged model does not reproduce the pre-merge output. The exact pre-merge weights
            are restored first, so the rollback does not depend on unmerge arithmetic being bit-exact.
    """
    from .error import MergeVerificationError

    previous_mode = base_model.training
    base_model.eval()
    snapshot = _base_tensor_snapshot(base_model)

    def rollback():
        if _any_layer_merged(base_model):
            unmerge_call()
        for _, saved, owner, attr in snapshot:
            with torch.no_grad():
                getattr(owner, attr).copy_(saved)

    try:
        inputs = _example_inputs(base_model)
        reference = None
        if inputs is not None:
            try:
                reference = _run_forward(base_model, inputs)
            except Exception:
                # Models whose forward needs more than `input_ids` (e.g. encoder-decoder models) fall back to the
                # exact weight comparison instead of reporting a spurious failure.
                reference = None

        if reference is None:
            _verify_by_round_trip(base_model, merge_call, unmerge_call, rollback)
            if keep_merge:
                # Merge again for real. The round trip left the model unmerged and its weights exactly as before; if
                # this second merge fails, the rollback that restores the snapshot also clears the partial merge.
                merge_call()
        else:
            try:
                merge_call()
            except BaseException:
                # A merge that fails half way must not leave a partially merged model behind.
                rollback()
                raise
            try:
                ok, reason = _outputs_close(reference, _run_forward(base_model, inputs))
                if not ok:
                    raise MergeVerificationError(
                        "Merge verification failed: the merged model does not reproduce the output of the adapter "
                        f"before merging ({reason}). The merge has been rolled back."
                    )
            except BaseException:
                rollback()
                raise

        if not keep_merge:
            # The trial merge is undone; the snapshot restore guarantees the caller sees an unmerged model.
            rollback()
    except BaseException:
        rollback()
        raise
    finally:
        base_model.train(previous_mode)


############################
# LOAD-TIME BINDING CHECKS #
############################


def clear_digest_cache(peft_model: torch.nn.Module) -> None:
    """Drop the memoized base digest after the base weights were changed in place (e.g. by a merge)."""
    try:
        delattr(unwrap_base_model(peft_model), _DIGEST_CACHE_ATTR)
    except AttributeError:
        pass


def current_binding(peft_model: torch.nn.Module) -> dict[str, Any]:
    """The binding record describing the base model as it is right now."""
    return compute_base_binding(unwrap_base_model(peft_model))


def verify_target_modules(
    recorded: dict[str, Any],
    peft_model: torch.nn.Module,
    adapter_name: str,
    *,
    force_load: bool,
) -> list[str]:
    """Check that every module the adapter was saved for exists on this model.

    Without this check, a module absent from the base model would be silently skipped while the adapter is injected.
    """
    from .error import BindingError

    recorded_modules = list(recorded.get("target_modules") or [])
    if not recorded_modules:
        return []
    base = unwrap_base_model(peft_model)
    # Any module of the unwrapped base model counts, including the wrapped originals inside tuner layers, which is
    # what the recorded names refer to (tuner infixes such as ``base_layer`` are stripped when recording).
    available = _base_module_names(base)
    missing = [name for name in recorded_modules if name not in available]
    if not missing:
        return []
    preview = ", ".join(repr(name) for name in missing[:5])
    suffix = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
    message = (
        f"Adapter {adapter_name!r} targets modules that do not exist in the base model: {preview}{suffix}. "
        "Loading it would silently skip those modules."
    )
    if force_load:
        warn(message)
        return missing
    raise BindingError(message + " Pass `force_load=True` to load it anyway.")


def prepare_binding_check(
    peft_model: torch.nn.Module,
    model_id: str | os.PathLike,
    adapter_name: str,
    *,
    force_load: bool,
    hf_hub_download_kwargs: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Read and check the binding record of an adapter that is about to be loaded.

    The base-model checks run before the adapter is injected, so a failed check never leaves a half-injected model.
    """
    from .error import BindingError

    recorded = read_binding(model_id, **(hf_hub_download_kwargs or {}))
    if recorded is not None and recorded.get("status") == "unavailable":
        message = (
            f"Adapter {adapter_name!r} has a {BINDING_NAME} that could not be read (the Hub is offline and the file is "
            "not cached). Its binding cannot be verified."
        )
        if force_load:
            warn(message + " Loading anyway because `force_load=True`.")
            return {"status": "unknown"}
        raise BindingError(message + " Pass `force_load=True` to load it anyway.")
    if recorded is None:
        message = (
            f"Adapter {adapter_name!r} has no {BINDING_NAME}, so its binding to this base model is unknown (adapters "
            "saved by older PEFT versions, or by `save_pretrained` without `save_binding=True`, have no such record)."
        )
        if force_load:
            warn(message + " Loading anyway because `force_load=True`.")
            return {"status": "unknown"}
        raise BindingError(message + " Pass `force_load=True` to load it anyway, or omit `verify_binding=True`.")

    if recorded.get("digest_algorithm") not in (None, DIGEST_ALGORITHM):
        # Records from a different digest algorithm cannot be compared; verify the identity fields only, and say so.
        warn(
            f"Adapter {adapter_name!r} records digest algorithm {recorded.get('digest_algorithm')!r}, but this PEFT "
            f"uses {DIGEST_ALGORITHM!r}. Only the base model name, revision, and target modules will be verified."
        )
        recorded = dict(recorded, base_model=dict(recorded.get("base_model") or {}, weight_digest=None))

    differences = compare_bindings(recorded, current_binding(peft_model))
    if differences:
        message = f"Adapter {adapter_name!r} was saved for a different base model: " + " | ".join(differences)
        if force_load:
            warn(message + " Loading anyway because `force_load=True`.")
        else:
            raise BindingError(message + ". Pass `force_load=True` to load it anyway.")

    verify_target_modules(recorded, peft_model, adapter_name, force_load=force_load)
    return recorded


def check_conflicts(
    peft_model: torch.nn.Module,
    adapter_name: str,
    recorded: dict[str, Any],
    *,
    force_load: bool,
) -> None:
    """Reject an adapter that targets the same modules as an adapter bound to a different base model."""
    from .error import BindingError

    if recorded.get("base_model") is None:
        return
    records = dict(getattr(peft_model, "_adapter_bindings", None) or {})
    records.pop(adapter_name, None)
    records[adapter_name] = recorded
    try:
        check_adapter_conflicts(records)
    except BindingError as exc:
        if force_load:
            warn(str(exc) + " Loading anyway because `force_load=True`.")
        else:
            raise
