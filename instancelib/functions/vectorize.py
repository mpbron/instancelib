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

import chunk
import itertools
from typing import Any, Iterator, List, Optional, Sequence, Tuple

import numpy.typing as npt

from ..environment import AbstractEnvironment
from ..feature_extraction import BaseVectorizer
from ..instances import Instance, InstanceProvider
from ..typehints.typevars import KT
from ..utils.numpy import matrix_tuple_to_vectors
from ..utils.to_key import to_key


def fit_vectorizer(
    vectorizer: BaseVectorizer[Instance[Any, Any, npt.NDArray[Any], Any]],
    provider: InstanceProvider[
        Instance[Any, Any, npt.NDArray[Any], Any],
        Any,
        Any,
        npt.NDArray[Any],
        Any,
    ],
    chunk_size: int = 200,
) -> BaseVectorizer[Instance[Any, Any, npt.NDArray[Any], Any]]:
    instances = list(
        itertools.chain.from_iterable(provider.instance_chunker(chunk_size))
    )
    vectorizer.fit(instances)
    return vectorizer


def vectorize_provider(
    vectorizer: BaseVectorizer[Instance[KT, Any, npt.NDArray[Any], Any]],
    provider: InstanceProvider[
        Instance[KT, Any, npt.NDArray[Any], Any],
        Any,
        Any,
        npt.NDArray[Any],
        Any,
    ],
    chunk_size: int = 200,
) -> Iterator[Tuple[Sequence[KT], Sequence[npt.NDArray[Any]]]]:
    instance_chunks = provider.instance_chunker(chunk_size)
    for instance_chunk in instance_chunks:
        matrix = vectorizer.transform(instance_chunk)
        keys: List[KT] = list(map(to_key, instance_chunk))  # type: ignore
        ret_keys, vectors = matrix_tuple_to_vectors(keys, matrix)
        yield ret_keys, vectors


def vectorize(
    vectorizer: BaseVectorizer[Instance[Any, Any, npt.NDArray[Any], Any]],
    environment: AbstractEnvironment[
        Instance[KT, Any, npt.NDArray[Any], Any],
        KT,
        Any,
        npt.NDArray[Any],
        Any,
        Any,
    ],
    fit: bool = True,
    chunk_size: int = 200,
    fit_instances: Optional[
        InstanceProvider[
            Instance[KT, Any, npt.NDArray[Any], Any],
            KT,
            Any,
            npt.NDArray[Any],
            Any,
        ]
    ] = None,
    transform_instances: Optional[
        InstanceProvider[
            Instance[KT, Any, npt.NDArray[Any], Any],
            KT,
            Any,
            npt.NDArray[Any],
            Any,
        ]
    ] = None,
    fit_chunk_size: Optional[int] = None,
    transform_chunk_size: Optional[int] = None,
):
    # Set parameters
    f_chunk_size = chunk_size if fit_chunk_size is None else fit_chunk_size
    t_chunk_size = (
        chunk_size if transform_chunk_size is None else transform_chunk_size
    )

    # Determine source and target provider
    source_provider = (
        fit_instances
        if fit_instances is not None
        else environment.all_instances
    )
    target_provider = (
        transform_instances
        if transform_instances is not None
        else environment.all_instances
    )

    # Vectorization Procedure
    if fit:
        vectorizer = fit_vectorizer(vectorizer, source_provider, f_chunk_size)
    results = vectorize_provider(vectorizer, target_provider, t_chunk_size)

    # Store the vectors in the Environment
    for keys, vecs in results:
        environment.add_vectors(keys, vecs)
