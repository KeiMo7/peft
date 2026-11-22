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


class PeftError(Exception):
    """Base PEFT error"""


class NoMatchingPeftModuleError(PeftError, ValueError):
    """The adapter being injected matched no module or parameter of the base model.

    Raised e.g. when the `target_modules` of the PEFT config matched nothing, which most often points at a
    misconfiguration. It subclasses `ValueError` for backwards compatibility with code that intercepted the generic
    error raised previously. Code that adds such an adapter intentionally can catch this error (or `PeftError`) and
    proceed.
    """


class BindingError(PeftError, ValueError):
    """The binding between an adapter and the base model could not be verified.

    Adapter binding verification is opt-in. It is triggered by `verify_binding=True` on
    [`~PeftModel.from_pretrained`] and [`~PeftModel.load_adapter`], and by `verify_merge=True` on the merge methods.
    The error is raised when the base model, the targeted modules, the adapter weights, or the bindings of coexisting
    adapters do not match the metadata recorded when the adapter was saved. Use `force_load=True` to downgrade these
    errors to warnings.
    """


class MergeVerificationError(BindingError):
    """An adapter merge was verified and did not reproduce the adapter's output.

    Raised by `merge_adapter(verify_merge=True)` and `merge_and_unload(verify_merge=True)`. The merge is rolled back
    before the error is raised, so the model is left in its pre-merge state.
    """
