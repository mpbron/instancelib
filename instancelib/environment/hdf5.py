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

from os import PathLike

from typing import Sequence, Iterable, Dict, Set

import numpy as np # type: ignore

from ..instances.hdf5 import HDF5BucketProvider, HDF5Provider 
from ..labels.memory import MemoryLabelProvider

from .base import AbstractEnvironment

# TODO Adjust MemoryEnvironment Generic Type (ADD ST)

class HDF5Environment(AbstractEnvironment[int, str, np.ndarray, str, str]):
    def __init__(
            self,
            dataset: HDF5Provider,
            labelprovider: MemoryLabelProvider[int, str],
        ):
        self._dataset = dataset
        self._labelprovider = labelprovider
        self._named_providers: Dict[str, HDF5Provider] = dict()

    @classmethod
    def from_data(cls, 
            target_labels: Iterable[str], 
            indices: Sequence[int], 
            data: Sequence[str], 
            ground_truth: Sequence[Set[str]],
            data_location: "PathLike[str]",
            vector_location: "PathLike[str]") -> HDF5Environment:
        dataset = HDF5Provider.from_data_and_indices(indices, data, data_location, vector_location)
        truth = MemoryLabelProvider[int, str].from_data(target_labels, indices, ground_truth)
        return cls(dataset, truth)

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
    def all_datapoints(self):
        return self._dataset

    @property
    def labels(self):
        return self._labelprovider

   


    
    
    



        

