from __future__ import annotations

from abc import ABC
from typing import (Any, Callable, Dict, FrozenSet, Generic, Iterable, Iterator, Mapping, MutableMapping,
                    Optional, Sequence, Tuple, TypeVar, Union)

from datasets.dataset_dict import DatasetDict

from .extractors import DataExtractor

from ..typehints.typevars import KT
from ..utils.func import invert_mapping, union

_T = TypeVar("_T")


def to_split_map(identifier_map: Mapping[KT, Tuple[str, int]]
                 ) -> Tuple[Mapping[str, Mapping[KT, int]], 
                            Mapping[str, Mapping[int, KT]]]:
    split_map = dict()
    for (key, (split, idx)) in identifier_map.items():
        split_map.setdefault(split, dict())[key] = idx
    inverse = {split: invert_mapping(mapping) for split, mapping in split_map.items()}
    return split_map, inverse

class HuggingFaceDataset(Mapping[KT, Mapping[str, Any]], Generic[KT]):
    splits: Sequence[str]
    dataset: DatasetDict
    identifier_map: Mapping[KT, Tuple[str, int]]
    split_map: Mapping[str, Mapping[KT, int]]
    inv_identifier_map: Mapping[str, Mapping[int, KT]]
    
    def __init__(self, 
                 dataset: DatasetDict,
                 splits: Sequence[str],
                 identifier_map: Mapping[KT, Tuple[str, int]],
                 split_map: Mapping[str, Mapping[KT, int]],
                 inv_identifier_map: Mapping[str, Mapping[int, KT]],
                 ) -> None:
        self.dataset = dataset
        self.splits = splits
        self.identifier_map = identifier_map
        self.split_map = split_map
        self.inv_identifier_map = inv_identifier_map
         
    def __iter__(self) -> Iterator[KT]:
        return iter(self.identifier_map)

    def __getitem__(self, __k: KT) -> Mapping[str, Any]:
        split, index = self.identifier_map[__k]
        data = self.dataset[split][index]
        return data
    
    def __len__(self) -> int:
        return len(self.identifier_map)

    @property
    def columns(self) -> Sequence[str]:
        if self.splits:
            first_split = self.splits[0]
            return self.dataset.column_names[first_split]
        return list()

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
              ) -> HuggingFaceDataset[KT]:
        split_map, inv_identifier_map = to_split_map(identifier_map)
        splits = tuple(split_map.keys())
        obj = cls(dataset, splits, identifier_map, split_map, inv_identifier_map)
        return obj

    @classmethod
    def from_other(cls, other: HuggingFaceDataset[KT], 
                        dataset: Optional[DatasetDict] = None) -> HuggingFaceDataset[KT]:
        chosen_dataset = other.dataset if dataset is None else dataset
        new_wrapper = HuggingFaceDataset(chosen_dataset, 
                                         other.splits, 
                                         other.identifier_map, 
                                         other.split_map, 
                                         other.inv_identifier_map)
        return new_wrapper

class HuggingFaceDatasetExtracted(Mapping[KT, _T], Generic[KT,_T]):
    hds: HuggingFaceDataset[KT]
    extractor: DataExtractor[_T]

    def __init__(self, hds: HuggingFaceDataset[KT], 
                       extractor: DataExtractor[_T]) -> None:
        self.hds = hds
        self.extractor = extractor

    def __iter__(self) -> Iterator[KT]:
        return iter(self.hds)

    def __getitem__(self, __k: KT) -> _T:
        return self.extractor(self.hds[__k])
    
    def __len__(self) -> int:
        return len(self.hds)

    def __contains__(self, __o: object) -> bool:
        return __o in self.hds