import dataclasses
from typing import Optional, Sequence, Type, TypeVar

from torch.utils._pytree import Context, PyTree, _register_pytree_node, tree_map
from typing_extensions import dataclass_transform

__all__ = ["register_dataclass_as_pytree", "dataclass_frozen_pytree", "tree_map"]


T = TypeVar("T")

def register_dataclass_as_pytree(Cls: Type[T], whitelist: Optional[Sequence[str]] = None) -> Type[T]:
    """Register a dataclass as a pytree, using the given whitelist of field names.

    :param Cls: The dataclass to register.
    :param whitelist: The names of the fields to include in the pytree. If None, all fields are included.
    :return: The dataclass, with the pytree registration applied. This is useful to be able to register a decorator.
    """

    assert dataclasses.is_dataclass(Cls)

    names = tuple(f.name for f in dataclasses.fields(Cls) if whitelist is None or f.name in whitelist)

    def flatten_fn(inst: Cls) -> tuple[list[PyTree], Context]:
        return list(getattr(inst, n) for n in names), names

    def unflatten_fn(values: list[PyTree], names: Context) -> Cls:
        return Cls(**dict(zip(names, values)))

    _register_pytree_node(Cls, flatten_fn, unflatten_fn)
    return Cls


@dataclass_transform()
def dataclass_frozen_pytree(Cls: Type, **kwargs) -> Type:
    """Decorator to make a frozen dataclass and register it as a PyTree."""
    dataCls = dataclasses.dataclass(frozen=True, slots=True, **kwargs)(Cls)
    register_dataclass_as_pytree(dataCls)
    return dataCls
