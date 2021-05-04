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

from ..instances import DataPointProvider, DataBucketProvider
from ..labels.memory import MemoryLabelProvider

from .base import AbstractEnvironment

from ..typehints import KT, DT, VT, RT, LT

# TODO Adjust MemoryEnvironment Generic Type (ADD ST)

class MemoryEnvironment(AbstractEnvironment[KT, DT, VT, RT, LT], Generic[KT, DT, VT, RT, LT]):
    def __init__(
            self,
            dataset: DataPointProvider[KT, DT, VT, RT],
            unlabeled: DataPointProvider[KT, DT, VT, RT],
            labeled: DataPointProvider[KT, DT, VT, RT],
            labelprovider: MemoryLabelProvider[KT, LT],
            truth: MemoryLabelProvider[KT, LT]
        ):
        self._dataset = dataset
        self._unlabeled = unlabeled
        self._labeled = labeled
        self._labelprovider = labelprovider
        self._named_providers: Dict[str, DataPointProvider[KT, DT, VT, RT]] = dict()
        self._storage: Dict[str, Any] = dict()
        self._truth = truth

    @classmethod
    def from_data(cls, 
            target_labels: Iterable[LT], 
            indices: Sequence[KT], 
            data: Sequence[DT], 
            ground_truth: Sequence[Iterable[LT]],
            vectors: Sequence[VT]) -> MemoryEnvironment[KT, DT, VT, RT, LT]:
        dataset = DataPointProvider[KT, DT, VT, RT].from_data_and_indices(indices, data, vectors)
        unlabeled = DataBucketProvider[KT, DT, VT](dataset, dataset.key_list)
        labeled = DataBucketProvider[KT, DT, VT](dataset, [])
        labels = MemoryLabelProvider[KT, LT].from_data(target_labels, indices, [])
        truth = MemoryLabelProvider[KT, LT].from_data(target_labels, indices, ground_truth)
        return cls(dataset, unlabeled, labeled, labels, truth)

    @classmethod
    def from_environment(cls, environment: AbstractEnvironment[KT, DT, VT, RT, LT], shared_labels: bool = True, *args, **kwargs) -> MemoryEnvironment[KT, DT, VT, RT, LT]:
        if isinstance(environment.dataset, DataPointProvider):
            dataset: DataPointProvider[KT, DT, VT, RT] = environment.dataset
        else:
            dataset = DataPointProvider[KT, DT, VT, RT].from_provider(environment.dataset)
        unlabeled = DataBucketProvider[KT, DT, VT](dataset, environment.unlabeled.key_list)
        labeled = DataBucketProvider[KT, DT, VT](dataset, environment.labeled.key_list)
        if isinstance(environment.labels, MemoryLabelProvider) and shared_labels:
            labels: MemoryLabelProvider[KT, LT] = environment.labels
        else:
            labels = MemoryLabelProvider[KT, LT](environment.labels.labelset, {}, {}) # type: ignore
        if isinstance(environment, MemoryEnvironment):                
            truth = environment.truth
        else:
            truth = MemoryLabelProvider[KT, LT](labels.labelset, {}, {})
        return cls(dataset, unlabeled, labeled, labels, truth)

    @classmethod
    def from_environment_only_data(cls, environment: AbstractEnvironment[KT, DT, VT, RT, LT]) -> MemoryEnvironment[KT, DT, VT, RT, LT]:
        if isinstance(environment.dataset, DataPointProvider):
            dataset: DataPointProvider[KT, DT, VT, RT] = environment.dataset
        else:
            dataset = DataPointProvider[KT, DT, VT, RT].from_provider(environment.dataset)
        unlabeled = DataBucketProvider[KT, DT, VT](dataset, environment.dataset.key_list)
        labeled = DataBucketProvider[KT, DT, VT](dataset, [])
        labels = MemoryLabelProvider[KT, LT](environment.labels.labelset, {}, {})
        if isinstance(environment, MemoryEnvironment):                
            truth = environment.truth
        else:
            truth = MemoryLabelProvider[KT, LT](labels.labelset, {}, {})
        return cls(dataset, unlabeled, labeled, labels, truth)

    def create_named_provider(self, name: str) -> DataPointProvider[KT, DT, VT, RT]:
        self._named_providers[name] = DataBucketProvider[KT, DT, VT](self._dataset, [])
        return self._named_providers[name]

    def get_named_provider(self, name: str) -> DataPointProvider[KT, DT, VT, RT]:
        if name in self._named_providers:
            self.create_named_provider(name)
        return self._named_providers[name]

    def create_empty_provider(self) -> DataPointProvider[KT, DT, VT, RT]:
        return DataPointProvider([])

    @property
    def dataset(self) -> DataPointProvider[KT, DT, VT, RT]:
        return self._dataset

    @property
    def unlabeled(self) -> DataPointProvider[KT, DT, VT, RT]:
        return self._unlabeled

    @property
    def labeled(self) -> DataPointProvider[KT, DT, VT, RT]:
        return self._labeled

    @property
    def labels(self) -> MemoryLabelProvider[KT, LT]:
        return self._labelprovider

    @property
    def truth(self) -> MemoryLabelProvider[KT, LT]:
        return self._truth

    def store(self, key: str, value: Any) -> None:
        self._storage[key] = value

    
    def storage_exists(self, key: str) -> bool:
        return key in self._storage

    
    def retrieve(self, key: str) -> Any:
        return self._storage[key]

    
    
    



        

