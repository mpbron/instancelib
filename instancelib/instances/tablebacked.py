# Copyright (C) 2021 The InstanceLib Authors. All Rights Reserved.

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 3 of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (Any, Callable, Generic, Iterator, List, Mapping,
                    MutableMapping, Optional, Sequence, Set, Tuple, TypeVar,
                    Union)

from ..typehints import DT, KT, MT, RT, VT
from .base import Instance, InstanceProvider
from .vectorstorage import VectorStorage
from .children import MemoryChildrenMixin
IT = TypeVar("IT", bound="Instance[Any, Any, Any, Any]")


class RowInstance(Mapping[str, Any], Instance[KT, Mapping[str,Any], VT, Mapping[str, Any]], Generic[IT, KT, DT, VT, RT, MT]):
    
    def __init__(self,
                 provider: "TableProvider[IT, KT, DT, VT, RT, MT]", 
                 data: Mapping[str, Any],
                 vector: Optional[VT] = None,
                 ) -> None:
        self._provider = provider
        self._data = data
        self._vector = vector

    def __getitem__(self, __k: str) -> Any:
        return self.data[__k]

    def __contains__(self, __o: object) -> bool:
        return __o in self.data

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    @property
    def columns(self) -> Sequence[str]:
        return list(self.data.keys())    

    @property
    def data(self) -> Mapping[str, Any]:
        return self._data # type: ignore
    
    @property
    def representation(self) -> Mapping[str, Any]:
        return self._data
        
    @property
    def vector(self) -> Optional[VT]:
        return self._vector # type: ignore

    @vector.setter
    def vector(self, value: Optional[VT]) -> None:
        if value is not None:
            self._vector = value
            self._provider.vectors[self.identifier] = value



class TableProvider(MemoryChildrenMixin[IT, KT, DT, VT, RT],
                    InstanceProvider[IT, KT, DT, VT, RT],
                    ABC, Generic[IT, KT, DT, VT, RT, MT]):
    
    columns: Sequence[str]
    storage: MutableMapping[KT, MutableMapping[str, Any]]
    builder: Callable[[InstanceProvider[IT, KT, DT, VT, RT], KT, Mapping[str, Any], Optional[VT]], IT]
    vectors: VectorStorage[KT, VT, MT]

    def __init__(self, 
                 storage: MutableMapping[KT, MutableMapping[str, Any]],
                 columns: Sequence[str],
                 vectors: VectorStorage[KT, VT, MT],
                 builder: Callable[[InstanceProvider[IT, KT, DT, VT, RT], KT, Mapping[str, Any], Optional[VT]], IT],
                 children: MutableMapping[KT, Set[KT]],
                 parents: MutableMapping[KT, KT]):
        
        self.storage = storage
        self.columns = columns
        self.vectors = vectors
        self.children = children
        self.parents = parents
        self.builder = builder
        self._decompose_taboo_keys = ["_identifier", "_vector"]

    def _decompose(self, ins: Instance[KT, DT, VT, RT]) -> Tuple[KT, MutableMapping[str, Any], Optional[VT]]:
        identifier = ins.identifier
        data = {key: value for (key,value) in ins.__dict__.items() if key not in self._decompose_taboo_keys}
        vector = ins.vector
        return identifier, data, vector

    def __iter__(self) -> Iterator[KT]:
        yield from self.storage.keys()

    def __getitem__(self, key: KT) -> IT:
        data = self.storage[key]
        vector = self.vectors[key]
        return self.builder(self, key, data, vector)

    def __setitem__(self, key: KT, value: IT) -> None:
        idx, data, vector = self._decompose(value)
        assert idx == key, f"Identifier -- Key mismatch: {idx} != {key}"
        self.storage[key] = data
        if vector is not None:
            self.vectors[key] = vector

    def __delitem__(self, key: KT) -> None:
        del self.storage[key]

    def __len__(self) -> int:
        return len(self.storage)

    def __contains__(self, key: object) -> bool:
        return key in self.storage

    @property
    def empty(self) -> bool:
        return not self.storage

    def get_all(self) -> Iterator[IT]:
        yield from list(self.values())

    def clear(self) -> None:
        self.storage = {}
       
    def bulk_get_vectors(self, keys: Sequence[KT]) -> Tuple[Sequence[KT], Sequence[VT]]:
        return self.vectors.get_vectors(keys)

    def bulk_get_all(self) -> List[IT]:
        return list(self.get_all())
   
    def construct(*args: Any, **kwargs: Any) -> IT:
        raise NotImplementedError
