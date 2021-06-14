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

from typing import Generic, Iterable, Dict, TypeVar, Any

from ..instances.base import Instance, InstanceProvider
from ..instances.memory import MemoryBucketProvider
from ..labels.memory import MemoryLabelProvider

from .base import AbstractEnvironment

from ..typehints import KT, DT, VT, RT, LT


InstanceType = TypeVar("InstanceType", bound="Instance[Any, Any, Any, Any]", covariant=True)

class AbstractMemoryEnvironment(
        AbstractEnvironment[InstanceType, KT, DT, VT, RT, LT],
        ABC, Generic[InstanceType, KT, DT, VT, RT, LT]):

    _public_dataset: InstanceProvider[InstanceType, KT, DT, VT, RT]
    _dataset: InstanceProvider[InstanceType, KT, DT, VT, RT]
    _labelprovider: MemoryLabelProvider[KT, LT]
    _named_providers: Dict[str, InstanceProvider[InstanceType, KT, DT, VT, RT]] = dict()

    @property
    def dataset(self) -> InstanceProvider[InstanceType, KT, DT, VT, RT]:
        return self._public_dataset

    @property
    def all_datapoints(self) -> InstanceProvider[InstanceType, KT, DT, VT, RT]:
        return self._dataset
    
    @property
    def labels(self) -> MemoryLabelProvider[KT, LT]:
        return self._labelprovider

    def create_bucket(self, keys: Iterable[KT]) -> InstanceProvider[InstanceType, KT, DT, VT, RT]:
        return MemoryBucketProvider[InstanceType, KT, DT, VT, RT](self._dataset, keys)

    def create_empty_provider(self) -> InstanceProvider[InstanceType, KT, DT, VT, RT]:
        return self.create_bucket([])

    def set_named_provider(self, name: str, value: InstanceProvider[InstanceType, KT, DT, VT, RT]):
        self._named_providers[name] = value

    def create_named_provider(self, name: str) -> InstanceProvider[InstanceType, KT, DT, VT, RT]:
        self._named_providers[name] = self.create_empty_provider()
        return self._named_providers[name]   
    
class MemoryEnvironment(
    AbstractMemoryEnvironment[InstanceType, KT, DT, VT, RT, LT],
        Generic[InstanceType, KT, DT, VT, RT, LT]):
    
    def __init__(
            self,
            dataset: InstanceProvider[InstanceType, KT, DT, VT, RT],
            labelprovider: MemoryLabelProvider[KT, LT]
        ):
        self._dataset = dataset
        self._public_dataset = MemoryBucketProvider[InstanceType, KT, DT, VT, RT](dataset, dataset.key_list)
        self._labelprovider = labelprovider
        self._named_providers = dict()

    
    

    



    

    
    
    



        

