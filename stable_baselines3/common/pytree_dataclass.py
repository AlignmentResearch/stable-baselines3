from typing import Callable, TypeVar

import optree as ot
from optree import PyTree as PyTree

__all__ = ["tree_flatten", "PyTree"]

T = TypeVar("T")

SB3_NAMESPACE = "stable-baselines3"


def tree_flatten(
    tree: ot.PyTree[T],
    is_leaf: Callable[[T], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = SB3_NAMESPACE
) -> tuple[list[T], ot.PyTreeSpec]:
    """optree.tree_flatten(...) but the default namespace is SB3_NAMESPACE"""
    return ot.tree_flatten(tree, is_leaf, none_is_leaf=none_is_leaf, namespace=namespace)
