import dataclasses
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import optree as ot
import torch as th
from optree import CustomTreeNode, PyTree
from typing_extensions import dataclass_transform

from stable_baselines3.common.type_aliases import TensorIndex
from stable_baselines3.common.utils import zip_strict

__all__ = [
    "PyTreeDataclass",
    "MutablePyTreeDataclass",
    "TensorTree",
    "tree_empty",
    "tree_flatten",
    "tree_index",
    "tree_map",
]

S = TypeVar("S")
T = TypeVar("T")
U = TypeVar("U")

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

    def tree_flatten(self) -> tuple[Sequence[T], None, tuple[str, ...]]:  # pytype: disable=invalid-annotation
        names = self._names()
        return tuple(getattr(self, n) for n in names), None, names

    @classmethod
    def tree_unflatten(cls, metadata: None, children: Sequence[T]) -> CustomTreeNode[T]:  # pytype: disable=invalid-annotation
        return cls(**dict(zip_strict(cls._names(), children)))


@dataclass_transform(frozen_default=True)  # pytype: disable=not-supported-yet
class PyTreeDataclass(_PyTreeDataclassBase[T], Generic[T], frozen=True):
    "Abstract class for immutable dataclass PyTrees"
    ...


@dataclass_transform(frozen_default=False)  # pytype: disable=not-supported-yet
class MutablePyTreeDataclass(_PyTreeDataclassBase[T], Generic[T], frozen=False):
    "Abstract class for mutable dataclass PyTrees"
    ...


# Manually expand the concrete type PyTree[th.Tensor] to make mypy happy.
# See links in https://github.com/metaopt/optree/issues/6, generic recursive types are not currently supported in mypy
TensorTree = Union[
    th.Tensor,
    Tuple["TensorTree", ...],
    Tuple[th.Tensor, ...],
    List["TensorTree"],
    List[th.Tensor],
    Dict[Any, "TensorTree"],
    Dict[Any, th.Tensor],
    CustomTreeNode[th.Tensor],
    PyTree[th.Tensor],
    PyTreeDataclass[th.Tensor],
    MutablePyTreeDataclass[th.Tensor],
]

ConcreteTensorTree = TypeVar("ConcreteTensorTree", bound=TensorTree)


@overload
def tree_flatten(
    tree: TensorTree,
    is_leaf: Callable[[TensorTree], bool] | None,
    *,
    none_is_leaf: bool = False,
    namespace: str = SB3_NAMESPACE,
) -> tuple[list[th.Tensor], ot.PyTreeSpec]:
    ...


@overload
def tree_flatten(
    tree: PyTree[T],
    is_leaf: Callable[[T], bool] | None,
    *,
    none_is_leaf: bool = False,
    namespace: str = SB3_NAMESPACE,
) -> tuple[list[T], ot.PyTreeSpec]:
    ...


def tree_flatten(tree, is_leaf=None, *, none_is_leaf=False, namespace=SB3_NAMESPACE):
    return ot.tree_flatten(tree, is_leaf, none_is_leaf=none_is_leaf, namespace=namespace)


@overload
def tree_map(
    func: Callable[..., th.Tensor],
    tree: ConcreteTensorTree,
    *rests: TensorTree,
    is_leaf: Callable[[TensorTree], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = SB3_NAMESPACE,
) -> ConcreteTensorTree:
    ...


@overload
def tree_map(  # pytype: disable=invalid-annotation
    func: Callable[..., U],
    tree: PyTree[T],
    *rests: Any,
    is_leaf: Callable[[T], bool] | None = None,
    none_is_leaf: bool = False,
    namespace: str = "",
) -> PyTree[U]:
    ...


def tree_map(func, tree, *rests, is_leaf=None, none_is_leaf=False, namespace=SB3_NAMESPACE):  # type: ignore
    return ot.tree_map(func, tree, *rests, is_leaf=is_leaf, none_is_leaf=none_is_leaf, namespace=namespace)


def tree_empty(tree: ot.PyTree, namespace: str = SB3_NAMESPACE) -> bool:
    flattened_state, _ = ot.tree_flatten(tree, namespace=namespace)
    return not bool(flattened_state)


def tree_index(
    tree: ConcreteTensorTree,
    idx: TensorIndex,
    *,
    is_leaf: None | Callable[[TensorTree], bool] = None,
    none_is_leaf: bool = False,
    namespace: str = SB3_NAMESPACE,
) -> ConcreteTensorTree:
    return ot.tree_map(lambda x: x[idx], tree, is_leaf=is_leaf, none_is_leaf=none_is_leaf, namespace=namespace)  # type: ignore
