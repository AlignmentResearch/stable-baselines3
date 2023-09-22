import dataclasses
import functools
from typing import Any, Callable, Optional, Sequence, Type, TypeVar

import optree as ot
from optree import CustomTreeNode
from optree import PyTree as PyTree
from typing_extensions import dataclass_transform

__all__ = [
    "PyTree",
    "dataclass_frozen_pytree",
    "register_dataclass_as_pytree",
    "tree_empty",
    "tree_flatten",
    "tree_index",
    "tree_map",
]

T = TypeVar("T")

SB3_NAMESPACE = "stable-baselines3"


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


def register_dataclass_as_pytree(
    Cls: Type[CustomTreeNode], whitelist: Optional[Sequence[str]] = None, namespace: str = SB3_NAMESPACE
):
    """Register a dataclass as a pytree, using the given whitelist of field names.

    :param Cls: The dataclass to register.
    :param whitelist: The names of the fields to include in the pytree. If None, all fields are included.
    :return: The dataclass, with the pytree registration applied. This is useful to be able to register a decorator.
    """

    assert dataclasses.is_dataclass(Cls)

    names = tuple(f.name for f in dataclasses.fields(Cls) if whitelist is None or f.name in whitelist)

    def flatten_fn(inst: CustomTreeNode[T]) -> tuple[Sequence[T], tuple[str, ...]]:
        return tuple(getattr(inst, n) for n in names), names

    def unflatten_fn(context: Any, values: T) -> CustomTreeNode[T]:
        return Cls(**dict(zip(names, values)))  # type: ignore

    ot.register_pytree_node(Cls, flatten_fn, unflatten_fn, namespace=namespace)

    Cls.__iter__ = lambda self: iter(getattr(self, n) for n in names)
    return Cls


@dataclass_transform()
def dataclass_frozen_pytree(Cls: Type, **kwargs) -> Type[ot.PyTree]:
    """Decorator to make a frozen dataclass and register it as a PyTree."""
    new_kwargs = dict(frozen=True, slots=True)
    new_kwargs.update(kwargs)
    dataCls = dataclasses.dataclass(**new_kwargs)(Cls)
    register_dataclass_as_pytree(dataCls)
    return dataCls
