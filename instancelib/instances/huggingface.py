from typing import FrozenSet, Generic, Mapping, Sequence
from datasets.dataset_dict import DatasetDict

from ..typehints.typevars import DT, KT
from .dataset import ReadOnlyDataset
from ..utils.func import union

class HuggingFaceDataset(ReadOnlyDataset[KT,DT], Generic[KT,DT]):
    identifier_map: Mapping[KT, str]
    index_map: Mapping[str, Mapping[KT, int]]

    
    def __init__(self, 
                 dataset: DatasetDict, 
                 key_column: str, 
                 data_column: str) -> None:
        self.dataset = dataset
        self.splits: Sequence[str] = tuple(dataset.keys())
        self.key_column = key_column
        self.data_column = data_column
        self.identifier_map = { key: split
            for split in self.splits for key in self.dataset[split][key_column]    
        }
        self.index_map = {split: {key: idx for idx, key in enumerate(self.dataset[split][self.key_column])} for split in self.splits}

    def __getitem__(self, __k: KT) -> DT:
        split = self.identifier_map[__k]
        index = self.index_map[split][__k]
        return self.dataset[split][index][self.data_column]        
    
    def __len__(self) -> int:
        return len(self.identifier_map)

    @property
    def identifiers(self) -> FrozenSet[KT]:
        return frozenset(self.identifier_map)

    def __contains__(self, __o: object) -> bool:
        return __o in self.identifier_map