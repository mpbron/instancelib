from __future__ import annotations

from ..utils.chunks import divide_iterable_in_lists
from .vectorstorage import VectorStorage

from ..typehints import KT, VT, MT

import numpy as np

from typing import Any, Generic, Sequence, MutableMapping, Callable, Iterator, Union, Tuple

class MemoryVectorStorage(VectorStorage[KT,VT,MT], Generic[KT, VT, MT]):
    from_matrix: Callable[[MT], Sequence[VT]]
    to_matrix: Callable[[Sequence[VT]], MT]
    
    
    def __init__(self, 
                 storage: MutableMapping[KT,VT], 
                 from_matrix: Callable[[MT], Sequence[VT]],
                 to_matrix: Callable[[Sequence[VT]], MT]) -> None:
        self.storage = storage
        self.to_matrix = to_matrix
        self.from_matrix = from_matrix

    def writeable(self) -> bool:
        return True

    def __getitem__(self, k: KT) -> VT:
        return self.storage[k]
    
    def __setitem__(self, k: KT, value: VT) -> None:
        self.storage[k] = value

    def __contains__(self, item: object) -> bool:
        return item in self.storage

    def __iter__(self) -> Iterator[KT]:
        return iter(self.storage)

    def add_bulk(self, input_keys: Sequence[KT], input_values: Sequence[VT]) -> None:
        for key, value in zip(input_keys, input_values):
            self.storage[key] = value
    
    def get_matrix(self, keys: Sequence[KT]) -> Tuple[Sequence[KT], MT]:
        vectors = [self.storage[key] for key in keys]
        matrix = self.to_matrix(vectors)
        return keys, matrix

    def add_bulk_matrix(self, input_keys: Sequence[KT], matrix: MT) -> None:
        vectors = self.from_matrix(matrix)
        for key, vector in zip(input_keys, vectors):
            self.storage[key] = vector

    def get_vectors(self, keys: Sequence[KT]) -> Tuple[Sequence[KT], Sequence[VT]]:
        """Return the vectors that correspond with the `keys` 

        Parameters
        ----------
        keys : Sequence[KT]
            A list of identifier keys

        Returns
        -------
        Tuple[Sequence[KT], Sequence[VT]]
            A tuple containing two lists:

                - A list with identifier (order may differ from `keys` argument)
                - A list with vectors 
        """        
        vectors = [self[key] for key in keys]
        return keys, vectors


    def matrices_chunker(self, chunk_size: int = 200) -> Iterator[Tuple[Sequence[KT], MT]]:
        chunks = divide_iterable_in_lists(list(self.keys()), chunk_size)
        for chunk in chunks:
            vectors = [self[key] for key in chunk]
            matrix = self.to_matrix(vectors)
            yield chunk, matrix

    def get_vectors_chunked(self, keys: Sequence[KT], chunk_size: int = 200) -> Iterator[Tuple[Sequence[KT], Sequence[VT]]]:
        """Return vectors in chunks of `chunk_size` containing the vectors requested in `keys`

        Parameters
        ----------
        keys : Sequence[KT]
            A list of identifier keys
        chunk_size : int, optional
            The size of the chunks, by default 200

        Yields
        -------
        Tuple[Sequence[KT], Sequence[VT]]
            A tuple containing two lists:

                - A list with identifiers (order may differ from `keys` argument)
                - A list with vectors 
        """        
        chunks = divide_iterable_in_lists(keys, chunk_size)
        for chunk in chunks:
            vectors = [self[key] for key in chunk]
            yield chunk, vectors

    def get_vectors_zipped(self, keys: Sequence[KT], chunk_size: int = 200) -> Iterator[Sequence[Tuple[KT, VT]]]:
        chunks = divide_iterable_in_lists(keys, chunk_size)
        for chunk in chunks:
            tuples = [(key, self[key]) for key in chunk]
            yield tuples

    def __enter__(self) -> VectorStorage[KT, VT, MT]:
        return self
    
    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        pass

    @classmethod
    def create(cls, 
               from_matrix: Callable[[MT], Sequence[VT]], 
               to_matrix: Callable[[Sequence[VT]], MT]) -> MemoryVectorStorage[KT, VT, MT]:
        storage = dict()
        return cls(storage, from_matrix, to_matrix)
    
    def reload(self) -> None:
        """Reload the index from disk
        """
        pass

class NumpyFromMatrix:
    def __call__(self, matrix: np.ndarray) -> Sequence[np.ndarray]:
        return list(matrix)

class NumpyToMatrix:
    def __call__(self, vectors: Sequence[np.ndarray]) -> np.ndarray:
        return np.vstack(vectors)

class NumpyMemoryStorage(MemoryVectorStorage[KT, np.ndarray, np.ndarray], Generic[KT]):
    def __init__(self, storage: MutableMapping[KT, np.ndarray]) -> None:
        from_matrix = NumpyFromMatrix()
        to_matrix = NumpyToMatrix()
        super().__init__(storage, from_matrix, to_matrix)

    @classmethod
    def create(cls) -> NumpyMemoryStorage[KT]:
        return cls(dict())