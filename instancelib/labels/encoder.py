from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (FrozenSet, Generic, Iterable, Iterator, List, Mapping,
                    Sequence, Tuple)

import numpy as np
import sklearn  # type: ignore

from ..typehints import LMT, LT, LVT, PMT
from ..utils.func import invert_mapping


class LabelEncoder(ABC, Generic[LT, LVT, LMT, PMT]):

    @abstractmethod
    def initialize(self, labels: Iterable[LT]) -> None:
        pass

    @abstractmethod
    def encode(self, labels: Iterable[LT]) -> LVT:
        raise NotImplementedError

    @abstractmethod
    def encode_batch(self, labelings: Iterable[Iterable[LT]]) -> LMT:
        raise NotImplementedError
    
    @abstractmethod
    def decode_vector(self, vector: LVT) -> FrozenSet[LT]:
        raise NotImplementedError

    @abstractmethod
    def decode_matrix(self, matrix: LMT) -> Sequence[FrozenSet[LT]]:
        raise NotImplementedError
    
    @abstractmethod
    def decode_proba_matrix(self, matrix: PMT) -> Sequence[FrozenSet[Tuple[LT, float]]]:
        raise NotImplementedError

    @abstractmethod
    def get_label_column_index(self, label: LT) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def labels(self) -> Sequence[LT]:
        raise NotImplementedError


class DictionaryEncoder(LabelEncoder[LT, np.ndarray, np.ndarray, np.ndarray], Generic[LT]):
    
    def __init__(self, mapping: Mapping[LT, int]):
        self.mapping = mapping
        self.inv_mapping = invert_mapping(self.mapping)
        self.labelset = frozenset(self.mapping.keys())
        self._labels = [lab for _, lab in sorted(self.inv_mapping.items())]

    def initialize(self, labels: Iterable[LT]) -> None:
        self.mapping = {
            label: idx for (idx, label) in enumerate(labels)
        }
        self.inv_mapping = invert_mapping(self.mapping)

    def encode(self, labels: Iterable[LT]) -> np.ndarray:
        result = np.array([self.mapping[lab] for lab in labels]) # type: ignore
        return result

    def encode_batch(self, labelings: Iterable[Iterable[LT]]) -> np.ndarray:
        encoded = tuple(map(self.encode, labelings))
        result = np.vstack(encoded)
        return result

    def decode_vector(self, vector: np.ndarray) -> FrozenSet[LT]:
        listed: List[int] = vector.tolist() # type: ignore
        result = frozenset([self.inv_mapping[enc] for enc in listed])
        return result

    def decode_matrix(self, matrix: np.ndarray) -> Sequence[FrozenSet[LT]]:
        listed: List[int] = matrix.tolist()
        result = [frozenset([self.inv_mapping[enc]]) for enc in listed]
        return result

    def decode_proba_matrix(self, matrix: np.ndarray) -> Sequence[FrozenSet[Tuple[LT, float]]]:
        prob_mat: List[List[float]] = matrix.tolist()
        label_list = self.labels
        labels = [
            frozenset(zip(label_list, prob_vec))
            for prob_vec in prob_mat
        ]
        return labels

    @property
    def labels(self) -> Sequence[LT]:
        return self._labels

    def get_label_column_index(self, label: LT) -> int:
        label_list = self.labels
        return label_list.index(label)

    @classmethod
    def from_list(cls, labels: Iterable[LT]) -> DictionaryEncoder[LT]:
        mapping = {lab: idx for idx, lab in enumerate(labels)}
        return cls(mapping)

    @classmethod
    def from_inv(cls, inv_mapping: Mapping[int, LT]) -> DictionaryEncoder[LT]:
        mapping = invert_mapping(inv_mapping)
        return cls(mapping)





class MultilabelDictionaryEncoder(DictionaryEncoder[LT], Generic[LT]):
    def encode(self, labels: Iterable[LT]) -> np.ndarray:
        def return_binary(lab: LT, labeling: FrozenSet[LT]) -> int:
            return lab in labeling
        labeling = frozenset(labels)
        result = np.array([return_binary(lab, labeling) for lab in self.labels]) # type: ignore
        return result
    
    def _decode_binary(self, listed_vector: List[int]) -> Iterator[LT]:
        for idx, included in enumerate(listed_vector):
            if included > 0:
                yield self.inv_mapping[idx]

    def decode_vector(self, vector: np.ndarray) -> FrozenSet[LT]:
        listed = vector.tolist()
        result = frozenset(self._decode_binary(listed))
        return result

    def decode_matrix(self, matrix: np.ndarray) -> Sequence[FrozenSet[LT]]:
        listed: List[List[int]] = matrix.tolist()
        result = [frozenset(self._decode_binary(vec)) for vec in listed]
        return result

    
    

class SklearnLabelEncoder(LabelEncoder[LT, np.ndarray, np.ndarray, np.ndarray], Generic[LT]):

    def __init__(self, encoder: sklearn.base.TransformerMixin, labels: Iterable[LT]) -> None:
        self.labelset = frozenset(labels)
        self.encoder = encoder
        if self.labelset:
            self._fit_label_encoder()

    def initialize(self, labels: Iterable[LT]) -> None:
        self.labelset = frozenset(labels)
        self._fit_label_encoder()
    
    def _fit_label_encoder(self) -> None:
        self.encoder.fit(list(self.labelset)) # type: ignore

    def encode(self, labels: Iterable[LT]) -> np.ndarray:
        return self.encoder.transform(list(set(labels))) # type: ignore

    def encode_batch(self, labelings: Iterable[Iterable[LT]]) -> np.ndarray:
        formatted = [next(iter(labeling)) for labeling in labelings]
        encoded: np.ndarray = self.encoder.transform(formatted) # type: ignore
        return encoded

    def decode_vector(self, vector: np.ndarray) -> FrozenSet[LT]:
        first_labeling: LT = self.encoder.inverse_transform(vector).tolist()[0] # type: ignore
        return frozenset([first_labeling])

    def decode_matrix(self, matrix: np.ndarray) -> Sequence[FrozenSet[LT]]:
        labelings: Iterable[LT] = self.encoder.inverse_transform(matrix).tolist() # type: ignore
        return [frozenset([labeling]) for labeling in labelings]

    def get_label_column_index(self, label: LT) -> int:
        label_list = self.labels
        return label_list.index(label)

    @property
    def labels(self) -> Sequence[LT]:
        labels: Sequence[LT] = self.encoder.classes_.tolist() # type: ignore
        return labels

    def decode_proba_matrix(self, matrix: np.ndarray) -> Sequence[FrozenSet[Tuple[LT, float]]]:
        prob_mat: List[List[float]] = matrix.tolist()
        label_list = self.labels
        labels = [
            frozenset(zip(label_list, prob_vec))
            for prob_vec in prob_mat
        ]
        return labels

class SklearnMultiLabelEncoder(SklearnLabelEncoder[LT], Generic[LT]):

    def _fit_label_encoder(self) -> None:
        self.encoder.fit(list(map(lambda x: {x}, self._target_labels))) # type: ignore

    def encode_batch(self, labelings: Iterable[Iterable[LT]]) -> np.ndarray:
        formatted = [frozenset(labeling) for labeling in labelings]
        encoded: np.ndarray = self.encoder.transform(formatted) # type: ignore
        return encoded

    def encode(self, labels: Iterable[LT]) -> np.ndarray:
        return self.encoder.transform([list(set(labels))]) # type: ignore

    def decode_matrix(self, matrix: np.ndarray) -> Sequence[FrozenSet[LT]]:
        labelings: Iterable[Iterable[LT]] = self.encoder.inverse_transform(matrix) # type: ignore
        return [frozenset(labeling) for labeling in labelings]

    def decode_vector(self, vector: np.ndarray) -> FrozenSet[LT]:
        first_labeling: Iterable[LT] = self.encoder.inverse_transform(vector).tolist()[0] # type: ignore
        return frozenset(first_labeling)

    