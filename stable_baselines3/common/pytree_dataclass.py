import dataclasses
import functools
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import optree as ot
import torch as th
from optree import CustomTreeNode
from optree import PyTree as PyTree
from typing_extensions import dataclass_transform

from stable_baselines3.common.utils import zip_strict

__all__ = [
    "PyTree",
    "PyTreeDataclass",
    "MutablePyTreeDataclass",
    "TensorTree",
    "tree_empty",
    "tree_flatten",
    "tree_index",
    "tree_map",
]

T = TypeVar("T")

SB3_NAMESPACE = "stable-baselines3"


# We need to inherit from `type(CustomTreeNode)` to prevent conflicts due to different-inheritance in metaclasses.
# - For some reason just inheriting from `typing._ProtocolMeta` does not get rid of that error.
# - Inheriting from `typing._GenericAlias` is impossible, as it's a `typing._Final` class.
class _PyTreeDataclassMeta(type(CustomTreeNode)):  # type: ignore
    """Metaclass to register dataclasses as PyTrees.

    Usage:
      class MyDataclass(metaclass=_DataclassPyTreeMeta):
        ...
    """

    currently_registering: ClassVar[Optional[type]] = None

    def __new__(mcs, name, bases, namespace, slots=True, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace)

        if dataclasses.is_dataclass(cls):
            if mcs.currently_registering is not None:
                assert mcs.currently_registering.__module__ == cls.__module__
                assert mcs.currently_registering.__qualname__ == cls.__qualname__
                mcs.currently_registering = None
                return cls
            else:
                assert cls.__name__ in ["PyTreeDataclass", "MutablePyTreeDataclass"] or issubclass(
                    cls, (PyTreeDataclass, MutablePyTreeDataclass)
                )
                mcs.currently_registering = cls
        else:
            mcs.currently_registering = cls

        if name != "_PyTreeDataclassBase":
            if name not in ["PyTreeDataclass", "MutablePyTreeDataclass"]:
                frozen = issubclass(cls, PyTreeDataclass)
                if frozen:
                    assert not issubclass(cls, MutablePyTreeDataclass)
                else:
                    assert issubclass(cls, MutablePyTreeDataclass)
            else:
                frozen = kwargs.pop("frozen")

            cls = dataclasses.dataclass(frozen=frozen, slots=slots, **kwargs)(cls)
            assert issubclass(cls, CustomTreeNode)
            ot.register_pytree_node_class(cls, namespace=SB3_NAMESPACE)
        return cls


class _PyTreeDataclassBase(CustomTreeNode[T], metaclass=_PyTreeDataclassMeta):
    _names_cache: ClassVar[Optional[Tuple[str, ...]]] = None

    @classmethod
    def _names(cls) -> Tuple[str, ...]:
        if cls._names_cache is None:
            names = cls._names_cache = tuple(f.name for f in dataclasses.fields(cls))
        else:
            names = cls._names_cache
        return names

    def __iter__(self):
        seq, _, _ = self.tree_flatten()
        return iter(seq)

    def tree_flatten(self) -> tuple[Sequence[T], None, tuple[str, ...]]:
        names = self._names()
        return tuple(getattr(self, n) for n in names), None, names

    @classmethod
    def tree_unflatten(cls, metadata: None, children: Sequence[T]) -> CustomTreeNode[T]:
        return cls(**dict(zip_strict(cls._names(), children)))


@dataclass_transform(frozen_default=True)
class PyTreeDataclass(_PyTreeDataclassBase[T], frozen=True):
    "Abstract class for immutable dataclass PyTrees"
    ...


@dataclass_transform(frozen_default=False)
class MutablePyTreeDataclass(_PyTreeDataclassBase[T], frozen=False):
    "Abstract class for mutable dataclass PyTrees"
    ...


# Manually expand the concrete type PyTree[th.Tensor] to make mypy happy.
# See links in https://github.com/metaopt/optree/issues/6, generic recursive types are not currently supported in mypy
TensorTree = Union[
    th.Tensor,
    Tuple["TensorTree", ...],
    List["TensorTree"],
    Dict[Any, "TensorTree"],
    CustomTreeNode[th.Tensor],
    PyTree[th.Tensor],
]


tree_flatten = functools.wraps(ot.tree_flatten)(functools.partial(ot.tree_flatten, namespace=SB3_NAMESPACE))
tree_map = functools.wraps(ot.tree_map)(functools.partial(ot.tree_map, namespace=SB3_NAMESPACE))


def tree_empty(tree: ot.PyTree, namespace: str = SB3_NAMESPACE) -> bool:
    flattened_state, _ = ot.tree_flatten(tree, namespace=namespace)
    return not bool(flattened_state)


def tree_index(
    tree: PyTree,
    idx: Any,
    *,
    is_leaf: None | Callable[[PyTree], bool] = None,
    none_is_leaf: bool = False,
    namespace: str = SB3_NAMESPACE,
) -> PyTree:
    return ot.tree_map(lambda x: x[idx], tree, is_leaf=is_leaf, none_is_leaf=none_is_leaf, namespace=namespace)
