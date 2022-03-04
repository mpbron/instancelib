from typing import Any, Generic, MutableMapping, Sequence, Set, TypeVar, Union

from ..typehints import DT, KT, RT, VT
from ..utils.to_key import to_key
from .base import Instance

IT = TypeVar("IT", bound="Instance[Any, Any, Any, Any]")
class MemoryChildrenMixin(MutableMapping[KT, IT], Generic[IT, KT, DT,VT, RT]):
    parents: MutableMapping[KT, KT]
    children: MutableMapping[KT, Set[KT]]
  
    def add_child(self, 
                  parent: Union[KT, Instance[KT, DT, VT, RT]], 
                  child:  Union[KT, Instance[KT, DT, VT, RT]]) -> None:
        parent_key: KT = to_key(parent)
        child_key: KT = to_key(child)
        assert parent_key != child_key
        if parent_key in self and child_key in self:
            self.children.setdefault(parent_key, set()).add(child_key)
            self.parents[child_key] = parent_key
        else:
            raise KeyError("Either the parent or child does not exist in this Provider")

    def get_children(self, 
                     parent: Union[KT, Instance[KT, DT, VT, RT]]) -> Sequence[IT]:
        parent_key: KT = to_key(parent)
        if parent_key in self.children:
            children = [self[child_key] for child_key in self.children[parent_key]]
            return children # type: ignore
        return []

    def get_children_keys(self, parent: Union[KT, Instance[KT, DT, VT, RT]]) -> Sequence[KT]:
        parent_key: KT = to_key(parent)
        if parent_key in self.children:
            return list(self.children[parent_key])
        return []

    def get_parent(self, child: Union[KT, Instance[KT, DT, VT, RT]]) -> IT:
        child_key: KT = to_key(child)
        if child_key in self.parents:
            parent_key = self.parents[child_key]
            parent = self[parent_key]
            return parent # type: ignore
        raise KeyError(f"The instance with key {child_key} has no parent")

    def discard_children(self, parent: Union[KT, Instance[KT, DT, VT, RT]]) -> None:
        parent_key: KT = to_key(parent)
        if parent_key in self.children:
            children = self.children[parent_key]
            self.children[parent_key] = set()
            for child in children:
                del self[child]
