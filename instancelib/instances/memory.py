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
from abc import ABC

from uuid import UUID, uuid4

from ..utils.func import filter_snd_none
from ..utils.to_key import to_key

import itertools
from typing import (Any, Dict, Generic, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, TypeVar, Union)

from .base import AbstractBucketProvider, Instance, InstanceProvider


from ..typehints import KT, DT, VT, RT

InstanceType = TypeVar("InstanceType", bound="Instance[Any, Any, Any, Any]")
class DataPoint(Instance[KT, DT, VT, RT], Generic[KT, DT, VT, RT]):

    def __init__(self, identifier: KT, data: DT, vector: Optional[VT], representation: RT) -> None:
        self._identifier = identifier
        self._data = data
        self._vector = vector
        self._representation = representation

    @property
    def data(self) -> DT:
        return self._data

    @property
    def representation(self) -> RT:
        return self._representation

    @property
    def identifier(self) -> KT:
        return self._identifier

    @identifier.setter
    def identifier(self, value: KT) -> None:
        self._identifier = value

    @property
    def vector(self) -> Optional[VT]:
        return self._vector

    @vector.setter
    def vector(self, value: Optional[VT]) -> None:  # type: ignore
        self._vector = value

    @classmethod
    def from_instance(cls, instance: Instance[KT, DT, VT, RT]):
        return cls(instance.identifier, instance.data, instance.vector, instance.representation)




class AbstractMemoryProvider(InstanceProvider[InstanceType, KT, DT, VT, RT], 
                             ABC, Generic[InstanceType, KT, DT, VT, RT]):
    
    dictionary: Dict[KT, InstanceType]
    children: Dict[KT, Set[KT]]
    parents: Dict[KT, KT]

    def __iter__(self) -> Iterator[KT]:
        yield from self.dictionary.keys()

    def __getitem__(self, key: KT) -> InstanceType:
        return self.dictionary[key]

    def __setitem__(self, key: KT, value: InstanceType) -> None:
        self.dictionary[key] = value  # type: ignore

    def __delitem__(self, key: KT) -> None:
        del self.dictionary[key]

    def __len__(self) -> int:
        return len(self.dictionary)

    def __contains__(self, key: object) -> bool:
        return key in self.dictionary

    @property
    def empty(self) -> bool:
        return not self.dictionary

    def get_all(self) -> Iterator[InstanceType]:
        yield from list(self.values())

    def clear(self) -> None:
        self.dictionary = {}
       
    def bulk_get_vectors(self, keys: Sequence[KT]) -> Tuple[Sequence[KT], Sequence[VT]]:
        vectors = [self[key].vector  for key in keys]
        ret_keys, ret_vectors = filter_snd_none(keys, vectors) # type: ignore
        return ret_keys, ret_vectors

    def bulk_get_all(self) -> List[InstanceType]:
        return list(self.get_all())
   
    def add_child(self, 
                  parent: Union[KT, InstanceType], 
                  child:  Union[KT, InstanceType]) -> None:
        parent_key: KT = to_key(parent)
        child_key: KT = to_key(child)
        assert parent_key != child_key
        if parent_key in self and child_key in self:
            self.children.setdefault(parent_key, set()).add(child_key)
            self.parents[child_key] = parent_key
        else:
            raise KeyError("Either the parent or child does not exist in this Provider")

    def get_children(self, 
                     parent: Union[KT, InstanceType]) -> Sequence[InstanceType]:
        parent_key: KT = to_key(parent)
        if parent_key in self.children:
            children = [self.dictionary[child_key] for child_key in self.children[parent_key]]
            return children # type: ignore
        return []

    def get_parent(self, child: Union[KT, InstanceType]) -> InstanceType:
        child_key: KT= to_key(child)
        if child_key in self.parents:
            parent_key = self.parents[child_key]
            parent = self.dictionary[parent_key]
            return parent # type: ignore
        raise KeyError(f"The instance with key {child_key} has no parent")



 
class DataPointProvider(AbstractMemoryProvider[DataPoint[Union[KT, UUID], DT, VT, RT], Union[KT, UUID], DT, VT, RT], 
                        Generic[KT, DT, VT, RT]):

    def __init__(self, 
                 datapoints: Iterable[DataPoint[Union[KT, UUID], DT, VT, RT]],
                    ) -> None:
        self.dictionary = {data.identifier: data for data in datapoints}
        self.children = dict()
        self.parents = dict()

    def create(self, *args: Any, **kwargs: Any):
        new_key = uuid4()
        new_instance = DataPoint[Union[KT, UUID], DT, VT, RT](new_key, *args, **kwargs)
        self.add(new_instance)
        return new_instance

    @classmethod
    def from_data_and_indices(cls,
                              indices: Sequence[KT],
                              raw_data: Sequence[DT],
                              vectors: Optional[Sequence[Optional[VT]]] = None) -> DataPointProvider[KT, DT, VT, RT]:
        if vectors is None or len(vectors) != len(indices):
            vectors = [None] * len(indices)
        datapoints = itertools.starmap(
            DataPoint[Union[KT, UUID], DT, VT, RT], zip(indices, raw_data, vectors, raw_data))
        return cls(datapoints)

    @classmethod
    def from_data(cls, raw_data: Sequence[DT]) -> DataPointProvider[KT, DT, VT, RT]:
        indices = range(len(raw_data))
        vectors = [None] * len(raw_data)
        datapoints = itertools.starmap(
            DataPoint[Union[KT, UUID], DT, VT, RT], zip(indices, raw_data, vectors, raw_data))
        return cls(datapoints)


class MemoryBucketProvider(AbstractBucketProvider[InstanceType, KT, DT, VT, RT], Generic[InstanceType, KT, DT, VT, RT]):
    def __init__(self, 
                 dataset: InstanceProvider[InstanceType, KT, DT, VT, RT], 
                 instances: Iterable[KT]):
        self._elements: Set[KT] = set(instances)
        self.dataset = dataset

    def _add_to_bucket(self, key: KT) -> None:
        self._elements.add(key)

    def _remove_from_bucket(self, key: KT) -> None:
        self._elements.discard(key)

    def _clear_bucket(self) -> None:
        self._elements = set()

    def _in_bucket(self, key: KT) -> bool:
        return key in self._elements
    
    def _len_bucket(self) -> int:
        return len(self._elements)

    @property
    def _bucket(self) -> Iterable[KT]:
        iterable = iter(self._elements)
        return iterable
    

    


   