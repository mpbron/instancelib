from abc import ABC, abstractmethod
from os import PathLike
from typing import (Any, Callable, Dict, FrozenSet, Generic, Iterable, Iterator,
                    Mapping, Sequence, Tuple, TypeVar)

import numpy as np
import pandas as pd

from instancelib.instances.hdf5vector import HDF5VectorStorage

from ..typehints import DT, KT, RT, VT
from ..utils.chunks import divide_iterable_in_lists
from .base import Instance
from .external import ExternalProvider
from .hdf5 import HDF5VectorInstanceProvider
from .memory import AbstractMemoryProvider

IT = TypeVar("IT", bound="Instance[Any, Any, Any, Any]")

class ReadOnlyDataset(Mapping[KT, DT], ABC, Generic[KT, DT],):
    @abstractmethod
    def __getitem__(self, __k: KT) -> DT:
        raise NotImplementedError

    def get_bulk(self, keys: Sequence[KT]) -> Sequence[DT]:
        return [self[key] for key in keys]

    @property
    @abstractmethod
    def identifiers(self) -> FrozenSet[KT]:
        raise NotImplementedError

    @abstractmethod
    def __contains__(self, __o: object) -> bool:
        return super().__contains__(__o)

    def __iter__(self) -> Iterator[KT]:
        return iter(self.identifiers)

class PandasDataset(ReadOnlyDataset[int, Any]):
    def __init__(self, df: pd.DataFrame, data_col: str) -> None:
        self.df = df
        self.data_col = data_col
        self.ids = range(0, len(self.df))
        self.fids = frozenset(self.ids)

    def __getitem__(self, __k: int) -> Any:
        data: Any = self.df.iloc[__k][self.data_col]
        return data
    
    def __len__(self) -> int:
        return len(self.df)

    @property
    def identifiers(self) -> FrozenSet[int]:
        return self.fids

    def __contains__(self, __o: object) -> bool:
        return __o in self.ids

    def get_bulk(self, keys: Sequence[int]) -> Sequence[Any]:
        data: Sequence[Any] = self.df.iloc[keys][self.data_col] # type: ignore
        return data

class ReadOnlyProvider(ExternalProvider[IT, KT, DT, np.ndarray, RT],
                       HDF5VectorInstanceProvider[IT, KT, DT, RT], 
                       Generic[IT, KT, DT, RT]):
    local_data: Dict[KT, DT]
    local_representation: Dict[KT, RT]
    
    def __init__(self, 
                 dataset: ReadOnlyDataset[KT, DT], 
                 vector_storage_location: "PathLike[str]",
                 from_data_builder: Callable[[KT, DT], IT]) -> None:
        self.vector_storage_location = vector_storage_location
        self.vectorstorage = HDF5VectorStorage(self.vector_storage_location)
        self.dataset = dataset
        self.instance_cache = dict()
        self.local_data = dict()
        self._stores = (self.local_data, self.dataset)
        self.from_data_builder = from_data_builder
        
    
    def build_from_external(self, k: KT) -> IT:
        data = self.dataset[k]
        ins = self.from_data_builder(k, data)
        return ins
        
    def update_external(self, ins: Instance[KT, DT, np.ndarray, RT]) -> None:
        self.local_data[ins.identifier] == ins.data
        self.local_representation[ins.identifier] == ins.representation

    def __getitem__(self, k: KT) -> IT:
        if k in self.local:
            return self.dictionary[k]
        if k in self.instance_cache:
            instance = self.instance_cache[k]
            return instance
        if k in self.dataset:
            instance = self.build_from_external(k)
            self.instance_cache[k] = instance
            return instance
        raise KeyError(f"Instance with key {k} is not present in this provider")

    def __contains__(self, item: object) -> bool:
        disjunction = any(map(lambda x: item in x, self._stores))
        return disjunction

    def _get_local_keys(self, keys: Iterable[KT]) -> FrozenSet[KT]:
        return frozenset(self.dictionary).intersection(keys)

    def _get_cached_keys(self, keys: Iterable[KT]) -> FrozenSet[KT]:
        return frozenset(self.instance_cache).intersection(keys)

    def _get_external_keys(self, keys: Iterable[KT]) -> FrozenSet[KT]:
        return frozenset(self.dataset).intersection(keys).difference(self.dictionary)

    def _cached_data(self, keys: Iterable[KT], batch_size: int = 200) -> Iterator[Sequence[Tuple[KT, DT]]]:
        chunks = divide_iterable_in_lists(keys, batch_size)
        c = self.instance_cache
        for chunk in chunks:
            yield [(k, c[k].data) for k in chunk]

    def _cached_instances(self, keys: Iterable[KT], batch_size: int = 200) -> Iterator[Sequence[IT]]:
        chunks = divide_iterable_in_lists(keys, batch_size)
        c = self.instance_cache
        for chunk in chunks:
            yield [c[k] for k in chunk]

    def _external_data(self, keys: Iterable[KT], batch_size: int = 200) -> Iterator[Sequence[Tuple[KT, DT]]]:
        chunks = divide_iterable_in_lists(keys, batch_size)
        for chunk in chunks:
            datas = self.dataset.get_bulk(chunk)
            result = list(zip(chunk, datas))
            yield result

    def _external_instances(self, keys: Iterable[KT], batch_size: int = 200) -> Iterator[Sequence[IT]]:
        chunks = divide_iterable_in_lists(keys, batch_size)
        for chunk in chunks:
            datas = self.dataset.get_bulk(chunk)
            vectors = self.vectorstorage.get_vectors_zipped(chunk)

    def __iter__(self) -> Iterator[KT]:
        return iter(frozenset(self.dataset).union(self.dictionary))            

    def data_chunker(self, batch_size: int = 200) -> Iterator[Sequence[Tuple[KT, DT]]]:
        yield from self.data_chunker_selector(self.key_list, batch_size)

    def data_chunker_selector(self, keys: Iterable[KT], batch_size: int = 200) -> Iterator[Sequence[Tuple[KT, DT]]]:
        keyset = frozenset(keys)
        local_keys = self._get_local_keys(keyset)
        yield from super().data_chunker_selector(local_keys, batch_size)
        remaining_keys = frozenset(keyset).difference(local_keys)
        cached_keys = self._get_cached_keys(remaining_keys)
        yield from self._cached_data(cached_keys)
        remaining_keys = remaining_keys.difference(cached_keys)
        external_keys = self._get_external_keys(remaining_keys)
        yield from self._external_data(external_keys)

    def instance_chunker(self, batch_size: int = 200) -> Iterator[Sequence[IT]]:
        return super().instance_chunker(batch_size)
    
    def instance_chunker_selector(self, keys: Iterable[KT], batch_size: int = 200) -> Iterator[Sequence[IT]]:
        keyset = frozenset(keys)
        local_keys = self._get_local_keys(keyset)
        yield from super().instance_chunker_selector(local_keys, batch_size)
        remaining_keys = frozenset(keyset).difference(local_keys)
        cached_keys = self._get_cached_keys(remaining_keys)
        yield from self._cached_instances(cached_keys)
        remaining_keys = remaining_keys.difference(cached_keys)
        external_keys = self._get_external_keys(remaining_keys)
        yield from (self._external_data(external_keys)

    @staticmethod
    def construct(*args: Any, **kwargs: Any) -> IT:
        raise NotImplementedError

    def create(self, *args: Any, **kwargs: Any) -> IT:
        raise NotImplementedError