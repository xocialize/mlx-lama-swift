"""Load Big-LaMa's generator weights from the Lightning best.ckpt without running foreign code.

The checkpoint pickles Lightning callbacks and omegaconf configs next to the tensors. A restricted
unpickler resolves only torch / numpy / collections / builtin containers and swaps every other
class for an inert stub, so neither pytorch_lightning nor omegaconf needs to be installed and the
pickle can't import anything else.
"""
import pickle
import types

import torch

_ALLOWED = ("torch", "numpy", "collections", "_codecs")
_BUILTINS = {"dict", "list", "tuple", "set", "frozenset", "int", "float", "str", "bytes",
             "bool", "complex", "slice", "bytearray", "object"}


class _Stub:
    def __init__(self, *a, **k): pass
    def __setstate__(self, s): pass


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "builtins" and name in _BUILTINS:
            return super().find_class(module, name)
        if module.split(".")[0] in _ALLOWED:
            return super().find_class(module, name)
        return type(name, (_Stub,), {})


_safe_pickle = types.ModuleType("pickle")
_safe_pickle.Unpickler = _Unpickler
_safe_pickle.load = pickle.load


def generator_state_dict(path):
    """{key: tensor} for the generator, with the `generator.` prefix stripped."""
    ck = torch.load(path, map_location="cpu", weights_only=False, pickle_module=_safe_pickle)
    return {k[len("generator."):]: v for k, v in ck["state_dict"].items() if k.startswith("generator.")}
