from abc import ABC, abstractmethod
from os import PathLike
from .hdf5vector import HDF5VectorStorage
from typing import Any, Generic, Iterable, Iterator, Optional, Sequence, Tuple, TypeVar
from .base import InstanceProvider, Instance
from .vectorstorage import VectorStorage

from ..typehints import KT, DT, VT, RT, MT

import numpy as np

IT = TypeVar("IT", bound="Instance[Any, Any, Any, Any]")

class ExternalVectorInstanceProvider(InstanceProvider[IT, KT, DT, VT, RT], 
                                     ABC, Generic[IT, KT, DT, VT, RT, MT]):
    
    vectorstorage: Optional[VectorStorage[KT, VT, MT]]

    @abstractmethod
    def load_vectors(self) -> VectorStorage[KT, VT, MT]:
        raise NotImplementedError

    @abstractmethod
    def bulk_add_vectors(self, keys: Sequence[KT], values: Sequence[VT]) -> None:
        raise NotImplementedError

    def bulk_get_vectors(self, keys: Sequence[KT]) -> Tuple[Sequence[KT], Sequence[VT]]:
        if self.vectorstorage is None:
            self.vectorstorage = self.load_vectors()
        ret_keys, vectors = self.vectorstorage.get_vectors(keys)
        return ret_keys, vectors

    def vector_chunker_selector(self, 
                                keys: Iterable[KT], 
                                batch_size: int = 200) -> Iterator[Sequence[Tuple[KT, VT]]]:
        if self.vectorstorage is None:
            self.vectorstorage = self.load_vectors()
        results = self.vectorstorage.get_vectors_zipped(list(keys), batch_size)
        return results

    def vector_chunker(self, batch_size: int = 200) -> Iterator[Sequence[Tuple[KT, VT]]]:
        results = self.vector_chunker_selector(self.key_list, batch_size)
        return results

class HDF5VectorInstanceProvider(ExternalVectorInstanceProvider[IT, KT, DT, np.ndarray, RT, np.ndarray]):

    vector_storage_location: "PathLike[str]"

    def load_vectors(self) -> HDF5VectorStorage[KT]:
        return HDF5VectorStorage[KT](self.vector_storage_location)

    def bulk_add_vectors(self, keys: Sequence[KT], values: Sequence[np.ndarray]) -> None:
        with HDF5VectorStorage[KT](self.vector_storage_location, "a") as writeable_storage:
            writeable_storage.add_bulk(keys, values)
        self.vectorstorage = self.load_vectors()