from abc import ABC, abstractmethod
from dataclasses import dataclass
from threading import local
from typing import (Any, Callable, FrozenSet, Generic, Iterable, Iterator,
                    Mapping, MutableMapping, Optional, Sequence, Tuple, TypeVar, Union)

import numpy as np
import pandas as pd

from instancelib.environment.memory import MemoryEnvironment
from instancelib.utils.func import union

from ..typehints import DT, KT, RT, VT
from ..utils.chunks import divide_iterable_in_lists
from .base import Instance, InstanceProvider
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
        self.ids = frozenset(range(0, len(self.df)))

    def __getitem__(self, __k: int) -> Any:
        data: Any = self.df.iloc[__k][self.data_col]
        return data
    
    def __len__(self) -> int:
        return len(self.df)

    @property
    def identifiers(self) -> FrozenSet[int]:
        return self.ids

    def __contains__(self, __o: object) -> bool:
        return __o in self.ids

    def get_bulk(self, keys: Sequence[int]) -> Sequence[Any]:
        data: Sequence[Any] = self.df.iloc[keys][self.data_col] # type: ignore
        return data

class RowInstance(Mapping[str, Any], Instance[KT, Mapping[str,Any], VT, Mapping[str, Any]], Generic[KT, VT]):
    def __init__(self, 
                 data: Mapping[str, Any],
                 vector: Optional[VT] = None,
                 ) -> None:
        self._data = data
        self._vector = vector

    def __getitem__(self, __k: str) -> Any:
        return self.data[__k]

    def __contains__(self, __o: object) -> bool:
        return __o in self.data

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    @property
    def columns(self) -> Sequence[str]:
        return list(self.data.keys())    

    @property
    def data(self) -> Mapping[str, Any]:
        return self._data # type: ignore
    
    @property
    def representation(self) -> Mapping[str, Any]:
        return self._data
        
    @property
    def vector(self) -> Optional[VT]:
        return self._vector # type: ignore

    @vector.setter
    def vector(self, value) -> None:
        self._vector = value

class ExternalStorage(Generic[KT,DT,RT]):
    data_storage:  Mapping[KT, DT]
    repr_storage:  Mapping[KT, RT]
    other_storage: Mapping[str, Mapping[KT,Any]]
    def __init__(self,
                 data: Mapping[KT, DT],
                 repr: Mapping[KT, RT],
                 other: Mapping[str, Mapping[KT, Any]]):
        pass


class ReadOnlyProvider(ExternalProvider[IT, KT, DT, np.ndarray, RT],
                       HDF5VectorInstanceProvider[IT, KT, DT, RT],
                       Generic[IT, KT, DT, RT]):

    local_data: InstanceProvider[IT, KT, DT, np.ndarray, RT]
    _stores: Sequence[Mapping[KT, Any]]

    def __init__(self, 
                 dataset: ReadOnlyDataset[KT, DT],
                 from_data_builder: Callable[[KT, DT], IT],
                 local_data: InstanceProvider[IT, KT, DT, np.ndarray, RT]
                 ) -> None:
        self.instance_cache = dict()
        self.dataset = dataset
        self._stores = (self.local_data,
                        self.instance_cache,
                        self.dataset)
        self.from_data_builder = from_data_builder
    
    def build_from_external(self, k: KT) -> IT:
        data = self.dataset[k]
        ins = self.from_data_builder(k, data)
        return ins

    def update_external(self, ins: Instance[KT, DT, np.ndarray, RT]) -> None:
        return super().update_external(ins)

    def __getitem__(self, k: KT) -> IT:
        if k in self.instance_cache:
            instance = self.instance_cache[k]
            return instance
        if k in self.local_data:
            instance = self.local_data[k]
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
        return frozenset(self.local_data).intersection(keys)

    def _get_cached_keys(self, keys: Iterable[KT]) -> FrozenSet[KT]:
        return frozenset(self.instance_cache).intersection(keys)

    def _get_external_keys(self, keys: Iterable[KT]) -> FrozenSet[KT]:
        return frozenset(self.dataset).intersection(keys)

    def _cached_data(self, keys: Iterable[KT], batch_size: int = 200) -> Iterator[Sequence[Tuple[KT, DT]]]:
        chunks = divide_iterable_in_lists(keys, batch_size)
        c = self.instance_cache
        for chunk in chunks:
            yield [(k, c[k].data) for k in chunk]

    def _local_data(self, keys: Iterable[KT], batch_size: int = 200) -> Iterator[Sequence[Tuple[KT, DT]]]:
        return self.local_data.data_chunker_selector(keys, batch_size)
        
    def _external_data(self, keys: Iterable[KT], batch_size: int = 200) -> Iterator[Sequence[Tuple[KT, DT]]]:
        chunks = divide_iterable_in_lists(keys, batch_size)
        for chunk in chunks:
            datas = self.dataset.get_bulk(chunk)
            result = list(zip(chunk, datas))
            yield result
    @property
    def _all_keys(self) -> FrozenSet[KT]:
        return union(*map(lambda x: frozenset(x.keys()), self._stores))
    
    @property
    def key_list(self) -> Sequence[KT]:
        return list(self._all_keys)
    
    def __iter__(self) -> Iterator[KT]:
        return iter(self.key_list)            

    def data_chunker(self, batch_size: int = 200) -> Iterator[Sequence[Tuple[KT, DT]]]:
        yield from self.data_chunker_selector(self.key_list)

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

    def construct(*args: Any, **kwargs: Any) -> IT:
        raise NotImplementedError

    def create(*args: Any, **kwargs: Any) -> IT:
        raise NotImplementedError