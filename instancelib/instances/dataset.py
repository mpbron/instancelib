from abc import ABC, abstractmethod
from threading import local

from build import Iterator
import pandas as pd
from .external import ExternalProvider
from ..typehints import KT, DT, VT, RT
from typing import Any, Callable, FrozenSet, Iterable, Sequence, TypeVar, Generic, Mapping, Tuple
from .base import Instance
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

    @property
    def identifiers(self) -> FrozenSet[int]:
        return self.ids

    def __contains__(self, __o: object) -> bool:
        return __o in self.ids

    def get_bulk(self, keys: Sequence[int]) -> Sequence[Any]:
        data: Sequence[Any] = self.df.iloc[keys][self.data_col] # type: ignore
        return data

class ReadOnlyProvider(ExternalProvider[IT, KT, DT, VT, RT], 
                       AbstractMemoryProvider[IT, KT, DT, VT, RT], 
                       Generic[IT, KT, DT, VT, RT]):

    def __init__(self, 
                 dataset: ReadOnlyDataset[KT, DT], 
                 from_data_builder: Callable[[KT, DT], IT],
                 instances: Iterable[IT] = list(),
                 ) -> None:
        AbstractMemoryProvider[IT, KT, DT, VT, RT].__init__(self, instances)
        self.dataset = dataset
        self.instance_cache = dict()
        self._stores = (self.dictionary,
                       self.instance_cache,
                       self.dataset)
        self.from_data_builder = from_data_builder
    
    def build_from_external(self, k: KT) -> IT:
        data = self.dataset[k]
        ins = self.from_data_builder(k, data)
        return ins

    def __getitem__(self, k: KT) -> IT:
        if k in self.dictionary:
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
        return frozenset(self.dataset).intersection(keys)

    def
    
    def data_chunker_selector(self, keys: Iterable[KT], batch_size: int = 200) -> Iterator[Sequence[Tuple[KT, DT]]]:
        keyset = frozenset(keys)
        local_keys = self._get_local_keys(keyset)
        yield from super().data_chunker_selector(local_keys)
        _