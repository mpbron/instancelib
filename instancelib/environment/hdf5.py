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

from typing import Generic, Sequence, TypeVar, Iterable, Dict, Any, Set

import numpy as np # type: ignore

from ..instances.hdf5 import HDF5BucketProvider, HDF5Instance, HDF5Provider 
from ..labels.memory import MemoryLabelProvider

from .base import AbstractEnvironment

# TODO Adjust MemoryEnvironment Generic Type (ADD ST)

class HDF5Environment(AbstractEnvironment[int, str, np.ndarray, str, str]):
    def __init__(
            self,
            dataset: HDF5Provider,
            unlabeled: HDF5Provider,
            labeled: HDF5Provider,
            labelprovider: MemoryLabelProvider[int, str],
            truth: MemoryLabelProvider[int, str]
        ):
        self._dataset = dataset
        self._unlabeled = unlabeled
        self._labeled = labeled
        self._labelprovider = labelprovider
        self._named_providers: Dict[str, HDF5Provider] = dict()
        self._storage: Dict[str, Any] = dict()
        self._truth = truth

    @classmethod
    def from_data(cls, 
            target_labels: Iterable[str], 
            indices: Sequence[int], 
            data: Sequence[str], 
            ground_truth: Sequence[Set[str]],
            data_location: str,
            vector_location: str) -> HDF5Environment:
        dataset = HDF5Provider.from_data_and_indices(indices, data, data_location, vector_location)
        unlabeled = HDF5BucketProvider(dataset, dataset.key_list)
        labeled = HDF5BucketProvider(dataset, [])
        labels = MemoryLabelProvider[int, str].from_data(target_labels, indices, [])
        truth = MemoryLabelProvider[int, str].from_data(target_labels, indices, ground_truth)
        return cls(dataset, unlabeled, labeled, labels, truth)
    @classmethod
    def from_environment(cls, 
                         environment: AbstractEnvironment[int, str, np.ndarray, str, str], 
                         data_location: str = "", vector_location: str = "", 
                         shared_labels: bool = True, *args, **kwargs) -> HDF5Environment:
        if isinstance(environment.dataset, HDF5Provider):
            dataset = environment.dataset
        else:
            dataset = HDF5Provider.from_provider(environment.dataset, data_location, vector_location)
        unlabeled = HDF5BucketProvider(dataset, environment.unlabeled.key_list)
        labeled = HDF5BucketProvider(dataset, environment.labeled.key_list)
        if isinstance(environment.labels, MemoryLabelProvider) and shared_labels:
            labels: MemoryLabelProvider[int, str] = environment.labels
        else:
            labels = MemoryLabelProvider[int, str](environment.labels.labelset, {}, {}) # type: ignore
        if isinstance(environment.truth, MemoryLabelProvider):                
            truth = environment.truth
        else:
            truth = MemoryLabelProvider[int, str](labels.labelset, {}, {})
        return cls(dataset, unlabeled, labeled, labels, truth)

    @classmethod
    def from_environment_only_data(cls, 
                                   environment: AbstractEnvironment[int, str, np.ndarray, str, str],
                                   data_location: str, vector_location: str) -> HDF5Environment:
        if isinstance(environment.dataset, HDF5Provider):
            dataset = environment.dataset
        else:
            dataset = HDF5Provider.from_provider(environment.dataset, data_location, vector_location)
        unlabeled = HDF5BucketProvider(dataset, environment.dataset.key_list)
        labeled = HDF5BucketProvider(dataset, [])
        labels = MemoryLabelProvider[int, str](environment.labels.labelset, {}, {}) # type: ignore
        if isinstance(environment.truth, MemoryLabelProvider):                
            truth = environment.truth
        else:
            truth = MemoryLabelProvider[int, str](labels.labelset, {}, {})
        return cls(dataset, unlabeled, labeled, labels, truth)

    def create_named_provider(self, name: str) -> HDF5Provider:
        self._named_providers[name] = HDF5BucketProvider(self._dataset, [])
        return self._named_providers[name]

    def get_named_provider(self, name: str) -> HDF5Provider:
        if name in self._named_providers:
            self.create_named_provider(name)
        return self._named_providers[name]

    def create_empty_provider(self) -> HDF5BucketProvider:
        return HDF5BucketProvider(self._dataset, [])

    @property
    def dataset(self):
        return self._dataset

    @property
    def unlabeled(self):
        return self._unlabeled

    @property
    def labeled(self):
        return self._labeled

    @property
    def labels(self):
        return self._labelprovider

    @property
    def truth(self):
        return self._truth

    def store(self, key: str, value: Any) -> None:
        self._storage[key] = value

    
    def storage_exists(self, key: str) -> bool:
        return key in self._storage

    def retrieve(self, key: str) -> Any:
        return self._storage[key]

    
    
    



        

