import pytest

from stable_baselines3.common.pytree_dataclass import (
    FrozenPyTreeDataclass,
    MutablePyTreeDataclass,
    tree_map,
)


@pytest.mark.parametrize("parent_class", (FrozenPyTreeDataclass, MutablePyTreeDataclass))
def test_slots(parent_class):
    class D(parent_class):
        a: int
        b: str

    d = D(4, "b")

    assert D.__slots__ == ("a", "b")
    assert d.__slots__ == ("a", "b")

    d2 = tree_map(lambda x: x * 2, d)

    assert isinstance(d2, D)
    assert d2.__slots__ == d.__slots__
