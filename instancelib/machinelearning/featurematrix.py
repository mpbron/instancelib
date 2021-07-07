from __future__ import annotations

from typing import Any, Generic, Iterator, Optional, Sequence

import numpy as np

from ..instances import InstanceProvider
from ..typehints import KT
from ..utils.chunks import divide_sequence
from ..utils.func import filter_snd_none, list_unzip


class FeatureMatrix(Generic[KT]):
    def __init__(self, keys: Sequence[KT], vectors: Sequence[Optional[np.ndarray]]):
        # Filter all rows with None as Vector
        filtered_keys, filtered_vecs = filter_snd_none(keys, vectors) # type: ignore
        self.matrix = np.vstack(filtered_vecs)
        self.indices: Sequence[KT] = filtered_keys

    def get_instance_id(self, row_idx: int) -> KT:
        return self.indices[row_idx]

    @classmethod
    def generator_from_provider_mp(cls, provider: InstanceProvider[Any, KT, Any, np.ndarray, Any], batch_size: int = 100) -> Iterator[FeatureMatrix[KT]]:
        for key_batch in divide_sequence(provider.key_list, batch_size):
            ret_keys, vectors = provider.bulk_get_vectors(key_batch)
            matrix = cls(ret_keys, vectors)
            yield matrix

    @classmethod
    def generator_from_provider(cls,
                                provider: InstanceProvider[Any, KT, Any, np.ndarray, Any],
                                batch_size: int = 100) -> Iterator[FeatureMatrix[KT]]:
        for tuple_batch in provider.vector_chunker(batch_size):
            keys, vectors = list_unzip(tuple_batch)
            matrix = cls(keys, vectors)
            yield matrix
