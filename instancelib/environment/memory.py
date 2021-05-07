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

from typing import Generic, Sequence, Iterable, Dict
import numpy as np # type: ignore

from ..instances.memory import DataPointProvider, DataBucketProvider
from ..labels.memory import MemoryLabelProvider

from .base import AbstractEnvironment

from ..typehints import KT, DT, VT, RT, LT

class MemoryEnvironment(AbstractEnvironment[KT, DT, VT, RT, LT], Generic[KT, DT, VT, RT, LT]):
    def __init__(
            self,
            dataset: DataPointProvider[KT, DT, VT, RT],
            labelprovider: MemoryLabelProvider[KT, LT]
        ):
        self._dataset = dataset
        self._public_dataset = DataBucketProvider[KT, DT, VT, RT](dataset, dataset.key_list)
        self._labelprovider = labelprovider
        self._named_providers: Dict[str, DataPointProvider[KT, DT, VT, RT]] = dict()
        
    @classmethod
    def from_data(cls, 
            target_labels: Iterable[LT], 
            indices: Sequence[KT], 
            data: Sequence[DT], 
            ground_truth: Sequence[Iterable[LT]],
            vectors: Sequence[VT]) -> MemoryEnvironment[KT, DT, VT, RT, LT]:
        dataset = DataPointProvider[KT, DT, VT, RT].from_data_and_indices(indices, data, vectors)
        truth = MemoryLabelProvider[KT, LT].from_data(target_labels, indices, ground_truth)
        return cls(dataset, truth)

    def set_named_provider(self, name: str, value: DataPointProvider[KT, DT, VT, RT]):
        self._named_providers[name] = value


    def create_named_provider(self, name: str) -> DataPointProvider[KT, DT, VT, RT]:
        self._named_providers[name] = self.create_empty_provider()
        return self._named_providers[name]

    def get_named_provider(self, name: str) -> DataPointProvider[KT, DT, VT, RT]:
        if name in self._named_providers:
            self.create_named_provider(name)
        return self._named_providers[name]

    def create_empty_provider(self) -> DataPointProvider[KT, DT, VT, RT]:
        return DataBucketProvider(self._dataset, [])

    @property
    def dataset(self) -> DataPointProvider[KT, DT, VT, RT]:
        return self._public_dataset
    @property
    def labels(self) -> MemoryLabelProvider[KT, LT]:
        return self._labelprovider
    
    
    



        

