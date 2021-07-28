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
from typing import Any, Iterator, Optional, Sequence, Union

import numpy as np
from numpy.lib.shape_base import split  # type: ignore
import pandas as pd  # type: ignore

from .base import Instance
from .external import ExternalProvider
from .hdf5 import HDF5VectorInstanceProvider
from .hdf5vector import HDF5VectorStorage
from .memory import DataPoint
from .text import TextInstance


class HDF5TextInstance(DataPoint[Union[int, str], str, np.ndarray, str], 
                    TextInstance[Union[int, str], np.ndarray]):
    def __init__(self, 
                 identifier: Union[int, str], 
                 data: str, 
                 vector: Optional[np.ndarray], 
                 representation: Optional[str] = None, 
                 tokenized: Optional[Sequence[str]] = None,
                 map_to_original: Optional[np.ndarray] = None,
                 split_marker: Optional[Any] = None,
                 external: Optional[ExternalProvider[Any, Union[int, str], str, np.ndarray, str]] = None) -> None:
        representation = data if representation is None else representation
        super().__init__(identifier, data, vector, representation)
        self._tokenized = tokenized
        self._map_to_original = map_to_original
        self._split_marker = split_marker
        self._external = external

    @property
    def map_to_original(self) -> Optional[np.ndarray]:
        return self._map_to_original

    @map_to_original.setter
    def map_to_original(self, value: Optional[np.ndarray]) -> None:
        self._map_to_original = value
        if self._external is not None:
            self._external.update_external(self)

    @property
    def split_marker(self) -> Optional[Any]:
        return self._split_marker 

    @split_marker.setter
    def split_marker(self, value: Any):
        self._split_marker = value
        if self._external is not None:
            self._external.update_external(self)
    
    @property
    def tokenized(self) -> Optional[Sequence[str]]:
        return self._tokenized
    
    @tokenized.setter
    def tokenized(self, value: Sequence[str]) -> None:
        self._tokenized = value
        if self._external is not None:
            self._external.update_external(self)

class HDF5TextProvider(HDF5VectorInstanceProvider[HDF5TextInstance, Union[int, str], str, str], 
                         ExternalProvider[HDF5TextInstance, Union[int, str], str, np.ndarray, str]):
    def __init__(self, data_storage: "PathLike[str]", 
                       vector_storage_location: "PathLike[str]",
                       hdf5_dataset: str,
                       id_col: str,
                       data_cols: Sequence[str]
                       ) -> None:
        self.instance_cache = {}
        self.hdf5_dataset = hdf5_dataset
        self.id_col = id_col
        self.data_cols: Sequence[str] = data_cols
        self.data_storage = data_storage
        self.vector_storage_location = vector_storage_location
        self.vectorstorage = HDF5VectorStorage[Union[int, str]](vector_storage_location)

    def build_from_external(self, k: Union[int, str]) -> HDF5TextInstance:
        if self.vectorstorage is None:
            self.vectorstorage = self.load_vectors()
        df = self.dataframe
        row = df[df[self.id_col] == k] # type: ignore
        vec = self.vectorstorage[k]
        data: str = " ".join([row[col] for col in self.data_cols]) # type: ignore
        ins = HDF5TextInstance(k, 
                               data, vec, data,
                               tokenized=None, 
                               map_to_original=None, 
                               split_marker=None, 
                               external=self)
        return ins

    def update_external(self, ins: Instance[Union[int, str], str, np.ndarray, str]) -> None:
        pass


    @property
    def dataframe(self) -> pd.DataFrame:
        df: pd.DataFrame = pd.read_hdf(self.data_storage, self.hdf5_dataset)  # type: ignore
        return df

    def __iter__(self) -> Iterator[Union[int, str]]:
        key_col = self.dataframe[self.id_col]
        for _, key in key_col.items():  # type: ignore
            yield int(key) # type: ignore

   

    def __setitem__(self, key: int, value: Instance[int, str, np.ndarray, str]) -> None:
        pass

    def __delitem__(self, key: int) -> None:
        pass

    def __len__(self) -> int:
        return len(self.dataframe)

    def __contains__(self, key: object) -> bool:
        df = self.dataframe
        return len(df[df[self.id_col] == key]) > 0 # type: ignore

    @property
    def empty(self) -> bool:
        return not self.dataframe

    def get_all(self):
        yield from list(self.values())

    def clear(self) -> None:
        pass

    
