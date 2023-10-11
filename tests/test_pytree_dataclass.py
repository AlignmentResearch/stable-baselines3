from dataclasses import FrozenInstanceError
from typing import Optional

import pytest

import stable_baselines3.common.pytree_dataclass as ptd


@pytest.mark.parametrize("ParentPyTreeClass", (ptd.FrozenPyTreeDataclass, ptd.MutablePyTreeDataclass))
def test_dataclass_mapped_have_slots(ParentPyTreeClass: type) -> None:
    """
    If after running `tree_map` the class still has __slots__ and they're the same, then the correct class (the one with
    __slots__) is what has been registered as a Pytree custom node.
    """

    class D(ParentPyTreeClass):
        a: int
        b: str

    d = D(4, "b")

    assert D.__slots__ == ("a", "b")
    assert d.__slots__ == ("a", "b")

    d2 = ptd.tree_map(lambda x: x * 2, d)

    assert d2.a == 8 and d2.b == "bb"

    assert isinstance(d2, D)
    assert d2.__slots__ == d.__slots__


@pytest.mark.parametrize("ParentPyTreeClass", (ptd.FrozenPyTreeDataclass, ptd.MutablePyTreeDataclass))
def test_dataclass_frozen_explicit(ParentPyTreeClass: type) -> None:
    class D(ParentPyTreeClass):
        a: int

    with pytest.raises(TypeError, match="You should not specify frozen= for descendants"):

        class D(ParentPyTreeClass, frozen=True):  # type: ignore  # noqa:F811
            a: int


@pytest.mark.parametrize("frozen", (True, False))
def test_dataclass_must_be_descendant(frozen: bool) -> None:
    """classes with metaclass _PyTreeDataclassMeta must be descendants of FrozenPyTreeDataclass or MutablePyTreeDataclass"""

    # First with arbitrary name
    with pytest.raises(TypeError):

        class D(ptd._PyTreeDataclassBase, frozen=frozen):  # type: ignore
            pass

    with pytest.raises(TypeError):

        class D(metaclass=ptd._PyTreeDataclassMeta, frozen=frozen):  # type: ignore  # noqa: F811
            pass

    with pytest.raises(TypeError, match="[^ ]* dataclass .* should inherit"):

        class D(ptd._PyTreeDataclassBase):  # type: ignore  # noqa: F811
            pass

    with pytest.raises(TypeError, match="[^ ]* dataclass .* should inherit"):

        class D(metaclass=ptd._PyTreeDataclassMeta):  # type: ignore  # noqa: F811
            pass

    # Then try to copy each of the reserved names:
    ## _PyTreeDataclassBase
    with pytest.raises(TypeError):

        class _PyTreeDataclassBase(ptd._PyTreeDataclassBase, frozen=frozen):  # type: ignore
            pass

    with pytest.raises(TypeError):

        class _PyTreeDataclassBase(metaclass=ptd._PyTreeDataclassMeta, frozen=frozen):  # type: ignore
            pass

    with pytest.raises(TypeError, match="You cannot have another class named"):

        class _PyTreeDataclassBase(ptd._PyTreeDataclassBase):  # type: ignore
            pass

    with pytest.raises(TypeError, match="You cannot have another class named"):

        class _PyTreeDataclassBase(metaclass=ptd._PyTreeDataclassMeta):  # type: ignore
            pass

    ## FrozenPyTreeDataclass
    with pytest.raises(TypeError):

        class FrozenPyTreeDataclass(ptd._PyTreeDataclassBase, frozen=frozen):  # type: ignore
            pass

    with pytest.raises(TypeError):

        class FrozenPyTreeDataclass(metaclass=ptd._PyTreeDataclassMeta, frozen=frozen):  # type: ignore  # noqa: F811
            pass

    with pytest.raises(TypeError, match="You cannot have another class named"):

        class FrozenPyTreeDataclass(ptd._PyTreeDataclassBase):  # type: ignore  # noqa: F811
            pass

    with pytest.raises(TypeError, match="You cannot have another class named"):

        class FrozenPyTreeDataclass(metaclass=ptd._PyTreeDataclassMeta):  # type: ignore  # noqa: F811
            pass

    ## MutablePyTreeDataclass
    with pytest.raises(TypeError):

        class MutablePyTreeDataclass(ptd._PyTreeDataclassBase, frozen=frozen):  # type: ignore
            pass

    with pytest.raises(TypeError):

        class MutablePyTreeDataclass(metaclass=ptd._PyTreeDataclassMeta, frozen=frozen):  # type: ignore  # noqa:F811
            pass

    with pytest.raises(TypeError, match="You cannot have another class named"):

        class MutablePyTreeDataclass(ptd._PyTreeDataclassBase):  # type: ignore  # noqa:F811
            pass

    with pytest.raises(TypeError, match="You cannot have another class named"):

        class MutablePyTreeDataclass(metaclass=ptd._PyTreeDataclassMeta):  # type: ignore  # noqa:F811
            pass


def test_dataclass_frozen_or_not() -> None:
    class MutA(ptd.MutablePyTreeDataclass):
        a: int

    class FrozenA(ptd.FrozenPyTreeDataclass):
        a: int

    inst1 = MutA(2)
    inst2 = FrozenA(2)

    inst1.a = 2
    with pytest.raises(FrozenInstanceError):
        inst2.a = 3  # type: ignore[misc]


@pytest.mark.parametrize("ParentPyTreeClass", (ptd.FrozenPyTreeDataclass, ptd.MutablePyTreeDataclass))
def test_dataclass_inheriting_dataclass(ParentPyTreeClass: type) -> None:
    class A(ParentPyTreeClass):
        a: int

    inst = A(3)
    assert inst.a == 3

    class B(A):
        b: int

    inst = B(2, 4)
    assert inst.a == 2
    assert inst.b == 4


def test_tree_flatten() -> None:
    class A(ptd.FrozenPyTreeDataclass):
        a: Optional[int]

    flat, _ = ptd.tree_flatten((A(3), A(None), {"a": A(4)}))  # type: ignore
    assert flat == [3, 4]


def test_tree_map() -> None:
    class A(ptd.FrozenPyTreeDataclass):
        a: Optional[int]

    out = ptd.tree_map(lambda x: x * 2, ([2, 3], 4, A(5), None, {"a": 6}))  # type: ignore
    assert out == ([4, 6], 8, A(10), None, {"a": 12})


def test_tree_empty() -> None:
    assert ptd.tree_empty(())  # type: ignore
    assert ptd.tree_empty([])  # type: ignore
    assert ptd.tree_empty({})  # type: ignore
    assert not ptd.tree_empty({"a": 2})  # type: ignore
    assert not ptd.tree_empty([2])  # type: ignore

    class A(ptd.FrozenPyTreeDataclass):
        a: Optional[int]

    assert ptd.tree_empty([A(None)])  # type: ignore
    assert not ptd.tree_empty([A(None)], none_is_leaf=True)  # type: ignore
    assert not ptd.tree_empty([A(2)])  # type: ignore


def test_tree_index() -> None:
    l1 = ["a", "b", "c"]
    l2 = ["hi", "bye"]
    idx = 1

    e1 = l1[idx]
    e2 = l2[idx]

    class A(ptd.FrozenPyTreeDataclass):
        a: str

    out_tree = ptd.tree_index([A(l1), A(l2), l1, (l2, {"a": l1})], idx, is_leaf=lambda x: x is l1 or x is l2)  # type: ignore
    assert out_tree == [A(e1), A(e2), e1, (e2, {"a": e1})]
