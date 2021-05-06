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

from ..utils.func import filter_snd_none

import itertools
from typing import (Generic, Iterable, Iterator, List, Optional, Sequence, Tuple)

from .base import Instance, InstanceProvider


from ..typehints import KT, DT, VT, RT


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

    @property
    def vector(self) -> Optional[VT]:
        return self._vector

    @vector.setter
    def vector(self, value: Optional[VT]) -> None:  # type: ignore
        self._vector = value

    @classmethod
    def from_instance(cls, instance: Instance[KT, DT, VT, RT]):
        return cls(instance.identifier, instance.data, instance.vector, instance.representation)


class DataPointProvider(InstanceProvider[KT, DT, VT, RT], Generic[KT, DT, VT, RT]):

    def __init__(self, datapoints: Iterable[DataPoint[KT, DT, VT, RT]]) -> None:
        self.dictionary = {data.identifier: data for data in datapoints}

    @classmethod
    def from_data_and_indices(cls,
                              indices: Sequence[KT],
                              raw_data: Sequence[DT],
                              vectors: Optional[Sequence[Optional[VT]]] = None):
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
            DataPoint[KT, DT, VT, RT], zip(indices, raw_data, vectors))
        return cls(datapoints)

    def __iter__(self) -> Iterator[KT]:
        yield from self.dictionary.keys()

    def __getitem__(self, key: KT) -> Instance[KT, DT, VT, RT]:
        return self.dictionary[key]

    def __setitem__(self, key: KT, value: Instance[KT, DT, VT, RT]) -> None:
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

    def get_all(self) -> Iterator[Instance[KT, DT, VT, RT]]:
        yield from list(self.values())

    def clear(self) -> None:
        self.dictionary = {}
       
    def bulk_get_vectors(self, keys: Sequence[KT]) -> Tuple[Sequence[KT], Sequence[VT]]:
        vectors = [self[key].vector  for key in keys]
        ret_keys, ret_vectors = filter_snd_none(keys, vectors) # type: ignore
        return ret_keys, ret_vectors

    def bulk_get_all(self) -> List[Instance[KT, DT, VT, RT]]:
        return list(self.get_all())

    @classmethod
    def train_test_split(cls, 
                         source: InstanceProvider[KT, DT, VT, RT], 
                         train_size: int) -> Tuple[InstanceProvider[KT, DT, VT, RT], InstanceProvider[KT, DT, VT, RT]]:
        
        source_keys = list(frozenset(source.key_list))
        
        # Randomly sample train keys
        train_keys = random.sample(source_keys, train_size)
        # The remainder should be used for testing        
        test_keys = frozenset(source_keys).difference(train_keys)
        
        train_provider = cls([DataPoint.from_instance(source[key]) for key in train_keys])
        test_provider = cls([DataPoint.from_instance(source[key]) for key in test_keys])
        return train_provider, test_provider


    


class DataBucketProvider(DataPointProvider[KT, DT, VT, RT], Generic[KT, DT, VT, RT]):
    def __init__(self, dataset: InstanceProvider[KT, DT, VT, RT], instances: Iterable[KT]):
        self._elements = set(instances)
        self.dataset = dataset

    def __iter__(self) -> Iterator[KT]:
        yield from self._elements

    def __getitem__(self, key: KT):
        if key in self._elements:
            return self.dataset[key]
        raise KeyError(
            f"This datapoint with key {key} does not exist in this provider")

    def __setitem__(self, key: KT, value: Instance[KT, DT, VT, RT]) -> None:
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
                         source: InstanceProvider[KT, DT, VT, RT], 
                         train_size: int) -> Tuple[InstanceProvider[KT, DT, VT, RT], InstanceProvider[KT, DT, VT, RT]]:
        source_keys = list(frozenset(source.key_list))
        
        # Randomly sample train keys
        train_keys = random.sample(source_keys, train_size)
        # The remainder should be used for testing        
        test_keys = frozenset(source_keys).difference(train_keys)
        
        train_provider = cls(source, train_keys)
        test_provider = cls(source, test_keys)
        return train_provider, test_provider


