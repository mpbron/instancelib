from abc import ABC, abstractmethod

import numpy as np

import sklearn # type: ignore
from typing import FrozenSet, Generic, Iterable, List, Sequence, Tuple
from ..typehints import LT, LVT, LMT, PMT


class LabelEncoder(ABC, Generic[LT, LVT, LMT, PMT]):
    
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

class SklearnLabelEncoder(LabelEncoder[LT, np.ndarray, np.ndarray, np.ndarray], Generic[LT]):

    def __init__(self, encoder: sklearn.base.TransformerMixin, labels: Iterable[LT]) -> None:
        self.labelset = frozenset(labels)
        self.encoder = encoder
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

    