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

import random

from uuid import UUID, uuid4

from ..utils.func import filter_snd_none
from ..utils.to_key import to_key

import itertools
from typing import (Any, Dict, Generic, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, Union)

from .base import Instance, InstanceProvider


from ..typehints import KT, DT, VT, RT


class DataPoint(Instance[Union[KT, UUID], DT, VT, RT], Generic[KT, DT, VT, RT]):

    def __init__(self, identifier: Union[KT, UUID], data: DT, vector: Optional[VT], representation: RT) -> None:
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
    def identifier(self) -> Union[KT, UUID]:
        return self._identifier

    @identifier.setter
    def identifier(self, value: Union[KT, UUID]) -> None:
        self._identifier = value

    @property
    def vector(self) -> Optional[VT]:
        return self._vector

    @vector.setter
    def vector(self, value: Optional[VT]) -> None:  # type: ignore
        self._vector = value

    @classmethod
    def from_instance(cls, instance: Instance[Union[KT, UUID], DT, VT, RT]):
        return cls(instance.identifier, instance.data, instance.vector, instance.representation)


class DataPointProvider(InstanceProvider[Union[KT, UUID], DT, VT, RT], Generic[KT, DT, VT, RT]):

    def __init__(self, datapoints: Iterable[DataPoint[KT, DT, VT, RT]]) -> None:
        self.dictionary = {data.identifier: data for data in datapoints}
        self.children: Dict[Union[KT, UUID], Set[Union[KT, UUID]]] = dict()
        self.parents: Dict[Union[KT, UUID], Union[KT, UUID]] = dict()


    @classmethod
    def from_data_and_indices(cls,
                              indices: Sequence[KT],
                              raw_data: Sequence[DT],
                              vectors: Optional[Sequence[Optional[VT]]] = None) -> DataPointProvider[KT, DT, VT, RT]:
        if vectors is None or len(vectors) != len(indices):
            vectors = [None] * len(indices)
        datapoints = itertools.starmap(
            DataPoint[KT, DT, VT, RT], zip(indices, raw_data, vectors, raw_data))
        return cls(datapoints)

    @classmethod
    def from_data(cls, raw_data: Sequence[DT]) -> DataPointProvider[KT, DT, VT, RT]:
        indices = range(len(raw_data))
        vectors = [None] * len(raw_data)
        datapoints = itertools.starmap(
            DataPoint[KT, DT, VT, RT], zip(indices, raw_data, vectors, raw_data))
        return cls(datapoints)

    def __iter__(self) -> Iterator[Union[KT, UUID]]:
        yield from self.dictionary.keys()

    def __getitem__(self, key: Union[KT, UUID]) -> Instance[Union[KT, UUID], DT, VT, RT]:
        return self.dictionary[key]

    def __setitem__(self, key: KT, value: Instance[Union[KT, UUID], DT, VT, RT]) -> None:
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

    def get_all(self) -> Iterator[Instance[Union[KT, UUID], DT, VT, RT]]:
        yield from list(self.values())

    def clear(self) -> None:
        self.dictionary = {}
       
    def bulk_get_vectors(self, keys: Sequence[KT]) -> Tuple[Sequence[KT], Sequence[VT]]:
        vectors = [self[key].vector  for key in keys]
        ret_keys, ret_vectors = filter_snd_none(keys, vectors) # type: ignore
        return ret_keys, ret_vectors

    def bulk_get_all(self) -> List[Instance[Union[KT, UUID], DT, VT, RT]]:
        return list(self.get_all())

    @classmethod
    def train_test_split(cls, 
                         source: InstanceProvider[Union[KT, UUID], DT, VT, RT], 
                         train_size: int) -> Tuple[
                             InstanceProvider[Union[KT, UUID], DT, VT, RT], 
                             InstanceProvider[Union[KT, UUID], DT, VT, RT]]:
        
        source_keys = list(frozenset(source.key_list))
        
        # Randomly sample train keys
        train_keys = random.sample(source_keys, train_size)
        # The remainder should be used for testing        
        test_keys = frozenset(source_keys).difference(train_keys)
        
        train_provider = cls([DataPoint.from_instance(source[key]) for key in train_keys])
        test_provider = cls([DataPoint.from_instance(source[key]) for key in test_keys])
        return train_provider, test_provider

    def add_child(self, 
                  parent: Union[KT, UUID, Instance[Union[KT, UUID], DT, VT, RT]], 
                  child:  Union[KT, UUID, Instance[Union[KT, UUID], DT, VT, RT]]) -> None:
        parent_key: Union[KT, UUID] = to_key(parent)
        child_key: Union[KT, UUID] = to_key(child)
        assert parent_key != child_key
        if parent_key in self and child_key in self:
            self.children.setdefault(parent_key, set()).add(child_key)
            self.parents[child_key] = parent_key
        else:
            raise KeyError("Either the parent or child does not exist in this Provider")

    def get_children(self, 
                     parent: Union[KT, UUID, Instance[Union[KT, UUID], DT, VT, RT]]) -> Sequence[Instance[Union[KT, UUID], DT, VT, RT]]:
        parent_key: Union[KT, UUID] = to_key(parent)
        if parent_key in self.children:
            children = [self.dictionary[child_key] for child_key in self.children[parent_key]]
            return children # type: ignore
        return []

    def get_parent(self, child: Union[KT, UUID, Instance[Union[KT, UUID], DT, VT, RT]]) -> Instance[Union[KT, UUID], DT, VT, RT]:
        child_key: Union[KT, UUID]= to_key(child)
        if child_key in self.parents:
            parent_key = self.parents[child_key]
            parent = self.dictionary[parent_key]
            return parent # type: ignore
        raise KeyError(f"The instance with key {child_key} has no parent")

    def _new_key(self) -> UUID:
        new_key = uuid4()
        while new_key in self:
            new_key = uuid4()
        return new_key

    def create(self, *args: Any, **kwargs: Any):
        new_instance = DataPoint[KT, DT, VT, RT](self._new_key(), *args, **kwargs)
        self.add(new_instance)
        return new_instance

    


class DataBucketProvider(DataPointProvider[KT, DT, VT, RT], Generic[KT, DT, VT, RT]):
    def __init__(self, 
                 dataset: InstanceProvider[Union[KT, UUID], DT, VT, RT], 
                 instances: Iterable[Union[KT, UUID]]):
        self._elements = set(instances)
        self.dataset = dataset

    def __iter__(self) -> Iterator[Union[KT, UUID]]:
        yield from self._elements

    def __getitem__(self, key: Union[KT, UUID]):
        if key in self._elements:
            return self.dataset[key]
        raise KeyError(
            f"This datapoint with key {key} does not exist in this provider")

    def __setitem__(self, key: Union[KT, UUID], value: Instance[Union[KT, UUID], DT, VT, RT]) -> None:
        self._elements.add(key)
        self.dataset[key] = value  # type: ignore

    def __delitem__(self, key: KT) -> None:
        self._elements.discard(key)

    def __len__(self) -> int:
        return len(self._elements)

    def __contains__(self, key: object) -> bool:
        return key in self._elements

    @property
    def empty(self) -> bool:
        return not self._elements

    @classmethod
    def train_test_split(cls, 
                         source: InstanceProvider[Union[KT, UUID], DT, VT, RT], 
                         train_size: int) -> Tuple[
                             InstanceProvider[Union[KT, UUID], DT, VT, RT], 
                             InstanceProvider[Union[KT, UUID], DT, VT, RT]]:
        source_keys = list(frozenset(source.key_list))
        
        # Randomly sample train keys
        train_keys = random.sample(source_keys, train_size)
        # The remainder should be used for testing        
        test_keys = frozenset(source_keys).difference(train_keys)
        
        train_provider = cls(source, train_keys)
        test_provider = cls(source, test_keys)
        return train_provider, test_provider
    
    def add_child(self, 
                  parent: Union[KT, UUID, Instance[Union[KT, UUID], DT, VT, RT]], 
                  child: Union[KT, UUID, Instance[Union[KT, UUID], DT, VT, RT]]) -> None:
        self.dataset.add_child(parent, child)

    def get_children(self, parent: Union[KT, UUID, Instance[Union[KT, UUID], DT, VT, RT]]) -> Sequence[Instance[Union[KT, UUID], DT, VT, RT]]:
        return self.dataset.get_children(parent)

    def get_parent(self, child: Union[KT, UUID, Instance[Union[KT, UUID], DT, VT, RT]]) -> Instance[Union[KT, UUID], DT, VT, RT]:
        return self.dataset.get_parent(child)

    def create(self, *args: Any, **kwargs: Any): # type: ignore
        new_instance = self.dataset.create(*args, **kwargs)
        self.add(new_instance)
        return new_instance


