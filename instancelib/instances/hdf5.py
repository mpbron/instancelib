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

import functools
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from ..utils.chunks import divide_iterable_in_lists
from .base import Instance, InstanceProvider
from .hdf5vector import HDF5VectorStorage


class HDF5Instance(Instance[int, str, np.ndarray, str]):
    def __init__(self, identifier: int, data: str, vector: Optional[np.ndarray], 
                 vector_storage: HDF5VectorStorage[int]) -> None:
        self._identifier = identifier
        self._data = data
        self._vectorstorage = vector_storage
        self._vector = vector

    @property
    def data(self) -> str:
        return self._data

    @property
    def representation(self) -> str:
        return self._data

    @property
    def identifier(self) -> int:
        return int(self._identifier)

    @identifier.setter
    def identifier(self, value: Union[int]) -> None:
        self._identifier = value

    

    @property
    def vector(self) -> Optional[np.ndarray]:
        if self._vector is None:
            if self._identifier in self._vectorstorage:
                self._vector = self._vectorstorage[self._identifier]
        return self._vector

    @vector.setter
    def vector(self, value: Optional[np.ndarray]) -> None:  # type: ignore
        if value is not None:
            self._vector = value
            with HDF5VectorStorage[int](self._vectorstorage.h5path, "a") as writeable_storage:
                writeable_storage[self.identifier] = value

    @classmethod
    def from_row(cls, vectorstorage: HDF5VectorStorage[int], 
                 row: Sequence[Union[int, str]]) -> HDF5Instance:
        return cls(row[0], row[1], None, vectorstorage)  # type: ignore


class HDF5Provider(InstanceProvider[HDF5Instance, int, str, np.ndarray, str]):
    def __init__(self, data_storage: "PathLike[str]", vector_storage_location: "PathLike[str]") -> None:
        self.data_storage = data_storage
        self.vector_storage_location = vector_storage_location
        self.vectors = HDF5VectorStorage[int](vector_storage_location)

    @property
    def dataframe(self) -> pd.DataFrame:
        return pd.read_hdf(self.data_storage, "datastorage")  # type: ignore

    @classmethod
    def from_data_and_indices(cls,
                              indices: Sequence[int],
                              raw_data: Sequence[str],
                              data_storage: "PathLike[str]",
                              vector_storage_location: "PathLike[str]"):
        assert len(indices) == len(raw_data)
        datapoints = zip(indices, raw_data)
        dataframe = pd.DataFrame(datapoints, columns=["key", "data"])  # type: ignore
        dataframe.to_hdf(data_storage, "datastorage")  # type: ignore
        return cls(data_storage, vector_storage_location)

    @classmethod
    def from_data(cls, raw_data: Sequence[str], data_storage_location: "PathLike[str]", 
                  vector_storage_location: "PathLike[str]") -> HDF5Provider:
        indices = list(range(len(raw_data)))
        return cls.from_data_and_indices(indices, raw_data, data_storage_location, vector_storage_location)

    def __iter__(self) -> Iterator[int]:
        key_col = self.dataframe["key"]
        for _, key in key_col.items():  # type: ignore
            yield int(key) # type: ignore

    def __getitem__(self, key: int) -> HDF5Instance:
        df = self.dataframe
        row = df[df.key == key] # type: ignore
        vector: Optional[np.ndarray] = None
        if key in self.vectors:
            vector = self.vectors[key]
        return HDF5Instance(key, row["data"].values[0], vector, self.vectors) # type: ignore

    def __setitem__(self, key: int, value: Instance[int, str, np.ndarray, str]) -> None:
        pass

    def __delitem__(self, key: int) -> None:
        pass

    def __len__(self) -> int:
        return len(self.dataframe)

    def __contains__(self, key: object) -> bool:
        df = self.dataframe
        return len(df[df.key == key]) > 0 # type: ignore

    @property
    def empty(self) -> bool:
        return not self.dataframe

    def get_all(self):
        yield from list(self.values())

    def clear(self) -> None:
        pass

    def data_chunker(self, batch_size: int):
        constructor = functools.partial(HDF5Instance.from_row, self.vectors)
        df = self.dataframe
        instance_df = df.apply(constructor, axis=1)  # type: ignore
        instance_list: List[HDF5Instance] = instance_df.tolist() # type: ignore
        return divide_iterable_in_lists(instance_list, batch_size)

    def bulk_add_vectors(self, keys: Sequence[int], values: Sequence[np.ndarray]) -> None:
        with HDF5VectorStorage[int](self.vector_storage_location, "a") as writeable_storage:
            writeable_storage.add_bulk(keys, values)
        self.vectors.reload()

    def bulk_get_vectors(self, keys: Sequence[int]):
        ret_keys, vectors = self.vectors.get_vectors(keys)
        return ret_keys, vectors

    def vector_chunker(self, batch_size: int):
        self.vectors.reload()
        results = self.vectors.vectors_chunker(batch_size)
        yield from results