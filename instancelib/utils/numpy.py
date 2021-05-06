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

from typing import Iterable, Optional, Sequence, Tuple, TypeVar, Union

from h5py._hl.dataset import Dataset # type: ignore

import numpy as np  # type: ignore

KT = TypeVar("KT")
VT = TypeVar("VT")

def slicer(matrix: Union[Dataset, np.ndarray], slices: Iterable[Tuple[int, Optional[int]]]) -> np.ndarray:
        def get_slices_1d(): # type: ignore
            for slice_min, slice_max in slices:
                if slice_max is not None:
                    yield matrix[slice_min:slice_max]
                else:
                    yield matrix[slice_min]
        def get_slices_2d(): # type: ignore
            for slice_min, slice_max in slices:
                if slice_max is not None:
                    yield matrix[slice_min:slice_max,:]
                else:
                    yield matrix[slice_min,:]
        dims = len(matrix.shape) #type: ignore
        if dims == 1:
            return np.hstack(list(get_slices_1d())) # type: ignore
        return np.vstack(list(get_slices_2d())) # type: ignore

def matrix_to_vector_list(matrix: np.ndarray) -> Sequence[np.ndarray]:
    def get_vector(index: int) -> np.ndarray:
        return matrix[index, :]
    n_rows = matrix.shape[0]
    rows = range(n_rows)
    return list(map(get_vector, rows))

def matrix_tuple_to_vectors(keys: Sequence[KT], 
                            matrix: np.ndarray
                           ) -> Tuple[Sequence[KT], Sequence[np.ndarray]]:
    return keys, matrix_to_vector_list(matrix)

def matrix_tuple_to_zipped(keys: Sequence[KT], 
                           matrix: np.ndarray) -> Sequence[Tuple[KT, np.ndarray]]:
    result = list(zip(keys, matrix_to_vector_list(matrix)))
    return result
