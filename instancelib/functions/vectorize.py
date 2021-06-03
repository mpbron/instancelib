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

import itertools
from typing import List, Any

import numpy as np  # type: ignore

from ..environment import AbstractEnvironment
from ..feature_extraction import BaseVectorizer
from ..instances import Instance
from ..utils.to_key import to_key
from ..utils.numpy import matrix_tuple_to_vectors

from ..typehints.typevars import KT

def vectorize(vectorizer: BaseVectorizer[Instance[Any, Any, np.ndarray, Any]], 
              environment: AbstractEnvironment[Instance[Any, Any, np.ndarray, Any], Any, Any, np.ndarray, Any, Any],
              fit: bool = True, 
              chunk_size: int = 200) -> None:
    def fit_vector() -> None:
        instances = list(
            itertools.chain.from_iterable(
                environment.all_datapoints.data_chunker(chunk_size)))
        vectorizer.fit(instances)
    def set_vectors() -> None:
        instance_chunks = environment.all_datapoints.data_chunker(chunk_size)
        for instance_chunk in instance_chunks:
            matrix = vectorizer.transform(instance_chunk)
            keys: List[KT] = list(map(to_key, instance_chunk)) # type: ignore
            ret_keys, vectors = matrix_tuple_to_vectors(keys, matrix)
            environment.add_vectors(ret_keys, vectors)
    if fit:
        fit_vector()
    set_vectors()
