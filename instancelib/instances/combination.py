from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from threading import local
from typing import (Any, Callable, Dict, FrozenSet, Generic, Iterable, Iterator, List,
                    Mapping, MutableMapping, Optional, Sequence, Set, Tuple, TypeVar, Union)
from uuid import UUID, uuid4

import numpy as np
import pandas as pd

from instancelib.environment.memory import MemoryEnvironment
from instancelib.utils.func import union

from ..typehints import DT, KT, RT, VT
from ..utils.chunks import divide_iterable_in_lists
from .base import Instance, InstanceProvider, ROInstanceProvider
from .external import ExternalProvider
from .hdf5 import HDF5VectorInstanceProvider
from .memory import AbstractMemoryProvider



class UpdateHookInstance(Instance[KT, DT, VT, RT], ABC, Generic[KT, DT, VT, RT]):
    _update_hook: "Optional[Callable[[UpdateHookInstance[KT, DT, VT, RT]], Any]]"

    def register_hook(self, 
                       update_hook: "Callable[[UpdateHookInstance[KT, DT, VT, RT]], Any]",
                       ) -> None:
        self._update_hook = update_hook

    def update_hook(self) -> None:
        if self._update_hook is not None:
            self._update_hook(self)

IT = TypeVar("IT", bound="UpdateHookInstance[Any, Any, Any, Any]")

class CombinationProvider(InstanceProvider[IT, KT, DT, VT, RT],
                          Generic[IT, KT, DT, VT, RT]):

    read_only: Mapping[UUID,ROInstanceProvider[IT, KT, DT, VT, RT]]
    writeable: Mapping[UUID,InstanceProvider[IT, KT, DT, VT, RT]]
    new_data:  Mapping[UUID,InstanceProvider[IT, KT, DT, VT, RT]]
    combined:  Mapping[UUID,ROInstanceProvider[IT, KT, DT, VT, RT]]
    index_map: MutableMapping[KT, UUID]

    @classmethod
    def createprovider(cls,
               read_only: Sequence[ROInstanceProvider[IT, KT, DT, VT, RT]],
               writeable: Sequence[InstanceProvider[IT, KT, DT, VT, RT]],
               new_data: InstanceProvider[IT, KT, DT, VT, RT],
               ) -> CombinationProvider[IT, KT, DT, VT, RT]:
        read_onlys = {uuid4(): prov for prov in read_only}
        writeables= {uuid4(): prov for prov in writeable}
        new_datas  = dict([(uuid4(), new_data)])
        index_map = dict() 
        return cls(read_onlys, writeables, new_datas, index_map)


    def __init__(self, 
                 read_only: Mapping[UUID, ROInstanceProvider[IT, KT, DT, VT, RT]],
                 writeable: Mapping[UUID, InstanceProvider[IT, KT, DT, VT, RT]],
                 new_data:  Mapping[UUID, InstanceProvider[IT, KT, DT, VT, RT]],
                 index_map: MutableMapping[KT, UUID]
                 ) -> None:
        self.read_only = read_only
        self.writeable = writeable
        self.new_data  = new_data
        self.index_map = index_map
        self.combined = {**self.read_only, **self.writeable, **self.new_data}

    def __getitem__(self, k: KT) -> IT:
        if k in self.index_map:
            prov_id = self.index_map[k]
            instance = self.combined[prov_id][k]
            return instance
        raise KeyError(f"Instance with key {k} is not present in this provider")

    def __setitem__(self, __k: KT, __v: IT) -> None:
        prov_id = self.index_map[__k]
        if prov_id in self.writeable:
            self.writeable[prov_id][__k] = __v
        elif prov_id in self.new_data:
            self.new_data[prov_id][__k] = __v
        else:
            new_prov_id = next(iter(self.new_data.keys()))
            self.index_map[__k] = new_prov_id
            self.new_data[new_prov_id][__k] = __v

    def __delitem__(self, __v: KT) -> None:
        del self.index_map[__v]

    def __contains__(self, item: object) -> bool:
        return item in self.index_map

    def _divide_keys(self, keys: Iterable[KT]) -> Mapping[UUID, FrozenSet[KT]]:
        splits: Dict[UUID, List[KT]] = dict()
        for key in keys:
            splits.setdefault(self.index_map[key], list()).append(key)
        result = {prov_id: frozenset(values) for prov_id, values in splits.items()}
        return result      

    @property
    def _all_keys(self) -> FrozenSet[KT]:
        return frozenset(self.index_map)
    
    @property
    def key_list(self) -> Sequence[KT]:
        return list(self._all_keys)
    
    def __iter__(self) -> Iterator[KT]:
        return iter(self.key_list)            

    def data_chunker(self, batch_size: int = 200) -> Iterator[Sequence[Tuple[KT, DT]]]:
        yield from self.data_chunker_selector(self.key_list, batch_size)

    def data_chunker_selector(self, keys: Iterable[KT], batch_size: int = 200) -> Iterator[Sequence[Tuple[KT, DT]]]:
        splitted = self._divide_keys(keys)
        for prov_id, prov_keys in splitted.items():
            yield from self.combined[prov_id].data_chunker_selector(prov_keys, batch_size)
    
    def vector_chunker_selector(self, keys: Iterable[KT], batch_size: int = 200) -> Iterator[Sequence[Tuple[KT, VT]]]:
        splitted = self._divide_keys(keys)
        for prov_id, prov_keys in splitted.items():
            yield from self.combined[prov_id].vector_chunker_selector(prov_keys, batch_size)

    def vector_chunker(self, batch_size: int = 200) -> Iterator[Sequence[Tuple[KT, VT]]]:
        yield from self.vector_chunker_selector(self.key_list, batch_size)

    def create(self, *args: Any, **kwargs: Any) -> IT:
        provider = next(iter(self.new_data.values()))
        return provider.create(*args, **kwargs)