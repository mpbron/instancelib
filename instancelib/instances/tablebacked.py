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
from typing import (Any, Callable, Generic, Iterable, Iterator, List, Mapping,
                    MutableMapping, Optional, Sequence, Set, Tuple, TypeVar,
                    Union)

from .combination import UpdateHookInstance

from .extractors import ColumnExtractor, DataExtractor
from .memoryvectorstorage import MemoryVectorStorage

from ..typehints import DT, KT, MT, RT, VT
from .base import Instance, InstanceProvider, ROInstanceProvider
from .vectorstorage import VectorStorage
from .children import MemoryChildrenMixin

IT = TypeVar("IT", bound="UpdateHookInstance[Any, Any, Any, Any]")


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

class TableInstance(MutableMapping[str, Any], 
                    UpdateHookInstance[KT, DT, VT, RT], 
                    Generic[IT, KT, DT, VT, RT, MT]):
    _data_extractor: DataExtractor[DT]
    _repr_extractor: DataExtractor[RT]
    
    def __init__(self,
                 identifier: KT,
                 data: MutableMapping[str, Any],
                 vector: Optional[VT] = None,
                 data_extractor: DataExtractor[DT] = ColumnExtractor("data"),
                 repr_extractor: DataExtractor[RT] = ColumnExtractor("data")
                 ) -> None:

        self._identifier = identifier
        self._data = data
        self._vector = vector
        self._data_extractor = data_extractor
        self._repr_extractor = repr_extractor
        self._delete_hook = None
        self._update_hook = None

    def __getitem__(self, __k: str) -> Any:
        return self._data[__k]

    def __contains__(self, __o: object) -> bool:
        return __o in self._data

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __setitem__(self, __k: str, __v: Any) -> None:
        self._data[__k] = __v
        self.update_hook()

    def __delitem__(self, __v: str) -> None:
        del self._data[__v]
        self.update_hook()

    def _safe_get(self, key: str) -> Optional[Any]:
        if key in self:
            return self[key]
        return None
    
    @property
    def identifier(self) -> KT:
        return self._identifier

    @property
    def columns(self) -> Sequence[str]:
        return list(self._data.keys())    

    @property
    def data(self) -> DT:
        return self._data_extractor(self._data)
    
    @property
    def representation(self) -> RT:
        return self._repr_extractor(self._data)
        
    @property
    def vector(self) -> Optional[VT]:
        return self._vector # type: ignore
    
    @vector.setter
    def vector(self, value: Optional[VT]) -> None:
        if value is not None:
            self._vector = value
            self.update_hook()

class TableProviderRO(ROInstanceProvider[IT, KT, DT, VT, RT], Generic[IT,KT, DT, VT, RT, MT]):
    columns: Sequence[str]
    storage: Mapping[KT, Mapping[str, Any]]
    builder: Callable[[KT, Mapping[str, Any], Optional[VT]], IT]
    vectors: VectorStorage[KT, VT, MT]


    def __init__(self, 
                 storage: MutableMapping[KT, MutableMapping[str, Any]],
                 columns: Sequence[str],
                 vectors: VectorStorage[KT, VT, MT],
                 builder: Callable[[KT, Mapping[str, Any], Optional[VT]], IT],
                 ):
        self.storage = storage
        self.columns = columns
        self.vectors = vectors
        self.builder = builder
        
    def _decompose(self, ins: TableInstance[IT, KT, DT, VT, RT, MT]) -> Tuple[KT, Mapping[str, Any], Optional[VT]]:
        return ins.identifier, ins._data, ins.vector

    def __iter__(self) -> Iterator[KT]:
        yield from self.storage.keys()

    def _get_vector(self, key: KT) -> Optional[VT]:
        if key in self.vectors:
            return self.vectors[key]
        return None

    def __getitem__(self, key: KT) -> IT:
        data = self.storage[key]
        vector = self._get_vector(key)
        ins = self.builder(key, data, vector)
        return ins

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
        self.storage = dict()
       
    def bulk_get_vectors(self, keys: Sequence[KT]) -> Tuple[Sequence[KT], Sequence[VT]]:
        return self.vectors.get_vectors(keys)

    def vector_chunker_selector(self, keys: Iterable[KT], batch_size: int = 200) -> Iterator[Sequence[Tuple[KT, VT]]]:
        return self.vectors.get_vectors_zipped(list(keys), batch_size)

    def bulk_get_all(self) -> List[IT]:
        return list(self.get_all())


class TableProvider(MemoryChildrenMixin[IT, KT, DT, VT, RT],
                    TableProviderRO[IT, KT, DT, VT, RT, MT],
                    InstanceProvider[IT, KT, DT, VT, RT],
                    Generic[IT, KT, DT, VT, RT, MT]):

    storage:  MutableMapping[KT, MutableMapping[str, Any]]
    builder: Callable[[InstanceProvider[IT, KT, DT, VT, RT], KT, Mapping[str, Any], Optional[VT]], IT]
    
    def __init__(self, 
                 storage: MutableMapping[KT, MutableMapping[str, Any]],
                 columns: Sequence[str],
                 vectors: VectorStorage[KT, VT, MT],
                 builder: Callable[[ROInstanceProvider[IT, KT, DT, VT, RT], KT, Mapping[str, Any], Optional[VT]], IT],
                 children: MutableMapping[KT, Set[KT]],
                 parents: MutableMapping[KT, KT]):
        
        self.storage = storage
        self.columns = columns
        self.vectors = vectors
        self.children = children
        self.parents = parents
        self.builder = builder

    def __getitem__(self, __k: KT) -> IT:
        ins = super().__getitem__(__k)
        ins.register_hook(self._update)
        return ins

    def _decompose(self, ins: TableInstance[IT, KT, DT, VT, RT, MT]) -> Tuple[KT, Mapping[str, Any], Optional[VT]]:
        return ins.identifier, ins._data, ins.vector

    def _update(self, ins: IT) -> None:
        raise NotImplementedError

    def __setitem__(self, key: KT, value: IT) -> None:
        assert isinstance(value, TableInstance)
        idx, data, vector = self._decompose(value)
        assert idx == key, f"Identifier -- Key mismatch: {idx} != {key}"
        assert isinstance(data, MutableMapping)
        self.storage[key] = data
        if vector is not None:
            self.vectors[key] = vector

    def __delitem__(self, key: KT) -> None:
        del self.storage[key]
   
    def construct(*args: Any, **kwargs: Any) -> IT:
        raise NotImplementedError

    def create(self, *args: Any, **kwargs: Any) -> IT:
        raise NotImplementedError

