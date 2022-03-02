from __future__ import annotations

from abc import ABC
from typing import Any, Callable, Dict, FrozenSet, Generic, Iterable, Mapping, Optional, Sequence, Tuple, TypeVar, Union
from datasets.dataset_dict import DatasetDict

from ..typehints.typevars import KT
from .dataset import ReadOnlyDataset
from ..utils.func import invert_mapping, union

_T = TypeVar("_T")


class DataExtractor(ABC, Generic[_T]):
    def __call__(self, row: Mapping[str, Any]) -> _T:
        raise NotImplementedError

class IdentityExtractor(DataExtractor[Mapping[str, Any]]):
    def __call__(self, row: Mapping[str, Any]) -> Mapping[str, Any]:
        return row

class ColumnExtractor(DataExtractor[_T], Generic[_T]):
    def __init__(self, column: str):
        self.column = column

    def __call__(self, row: Mapping[str, Any]) -> _T:
        return row[self.column]

class ConcatenationExtractor(DataExtractor[str]):
    def __init__(self, *columns: str, seperator: str = " "):
        self.columns = list(columns)
        self.seperator = seperator

    def __call__(self, row: Mapping[str, Any]) -> str:
        return self.seperator.join((row[col] for col in self.columns))

class SubsetExtractor(IdentityExtractor):
    def __init__(self, *columns: str):
        self.columns = tuple(columns)

    def __call__(self, row: Mapping[str, Any]) -> Mapping[str, Any]:
        return {col: row[col] for col in self.columns}

def to_split_map(identifier_map: Mapping[KT, Tuple[str, int]]
                 ) -> Tuple[Mapping[str, Mapping[KT, int]], 
                                                                        Mapping[str, Mapping[int, KT]]]:
    split_map = dict()
    for (key, (split, idx)) in identifier_map.items():
        split_map.setdefault(split, dict())[key] = idx
    inverse = {split: invert_mapping(mapping) for split, mapping in split_map.items()}
    return split_map, inverse

class HuggingFaceDataset(ReadOnlyDataset[KT,_T], Generic[KT,_T]):
    splits: Sequence[str]
    dataset: DatasetDict
    identifier_map: Mapping[KT, Tuple[str, int]]
    split_map: Mapping[str, Mapping[KT, int]]
    inv_identifier_map: Mapping[str, Mapping[int, KT]]
    
    def __init__(self, 
                 dataset: DatasetDict,
                 extractor: Union[DataExtractor[_T], Callable[[Mapping[str, Any]], _T]],
                 splits: Sequence[str],
                 identifier_map: Mapping[KT, Tuple[str, int]],
                 split_map: Mapping[str, Mapping[KT, int]],
                 inv_identifier_map: Mapping[str, Mapping[int, KT]],
                 ) -> None:
        self.dataset = dataset
        self.splits = splits
        self.data_extractor = extractor
        self.identifier_map = identifier_map
        self.split_map = split_map
        self.inv_identifier_map = inv_identifier_map
         

    def __getitem__(self, __k: KT) -> _T:
        split, index = self.identifier_map[__k]
        data = self.data_extractor(self.dataset[split][index])
        return data
    
    def __len__(self) -> int:
        return len(self.identifier_map)

    @property
    def identifiers(self) -> FrozenSet[KT]:
        return frozenset(self.identifier_map)

    def __contains__(self, __o: object) -> bool:
        return __o in self.identifier_map

    @classmethod
    def get_identifier_map_from_column(cls, 
                               dataset: DatasetDict, 
                               key_column: str, 
                               splits: Iterable[str] = list()
                               ) -> Mapping[KT, Tuple[str, int]]:
        chosen_splits = tuple(splits) if splits else tuple(dataset.keys())
        identifier_map = { key: (split, idx)
            for split in chosen_splits for idx, key in enumerate(dataset[split][key_column])    
        }
        return identifier_map

    @classmethod
    def get_identifier_map_from_index(cls, 
                              dataset: DatasetDict,
                              splits: Iterable[str] = list()
                              ) -> Mapping[str, Tuple[str, int]]:
        chosen_splits: Sequence[str] = tuple(splits) if splits else tuple(dataset.keys())
        identifier_map = { f"{split}_{idx}": (split, idx)
            for split in chosen_splits for idx in range(len(dataset[split]))    
        }
        return identifier_map

    @classmethod
    def build(cls, 
              dataset: DatasetDict,
              identifier_map:  Mapping[KT, Tuple[str, int]],
              extractor: Union[DataExtractor[_T], Callable[[Mapping[str, Any]], _T]],
              ) -> HuggingFaceDataset[KT, _T]:
        split_map, inv_identifier_map = to_split_map(identifier_map)
        splits = tuple(split_map.keys())
        obj = cls(dataset, extractor, splits, identifier_map, split_map, inv_identifier_map)
        return obj

    @classmethod
    def from_other(cls, other: HuggingFaceDataset[KT, _T], 
                        extractor: Union[DataExtractor[_T], Callable[[Mapping[str, Any]], _T]],
                        dataset: Optional[DatasetDict] = None) -> HuggingFaceDataset[KT, _T]:
        new_wrapper = HuggingFaceDataset(other.dataset if dataset is None else dataset, 
                                         extractor,
                                         other.splits, 
                                         other.identifier_map, 
                                         other.split_map, 
                                         other.inv_identifier_map)
        return new_wrapper