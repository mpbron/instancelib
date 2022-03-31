from __future__ import annotations

from os import PathLike
from typing import (Any, FrozenSet, Generic, Iterable, Iterator, Optional,
                    Sequence, Tuple, TypeVar, Union)

import numpy as np

from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline

from ..feature_extraction.base import BaseVectorizer
from ..instances import Instance
from ..instances.base import InstanceProvider
from ..labels.encoder import LabelEncoder
from ..typehints import KT, LT
from ..utils.func import list_unzip
from .skvectors import SkLearnVectorClassifier

IT = TypeVar("IT", bound="Instance[Any, Any, Any, Any]", covariant=True)

class AutoVectorizerClassifier(SkLearnVectorClassifier[IT, KT, LT], Generic[IT, KT, LT]):
    def __init__(self, 
                 estimator: Union[ClassifierMixin, Pipeline], 
                 vectorizer: BaseVectorizer[IT],
                 encoder: LabelEncoder[LT, np.ndarray, np.ndarray, np.ndarray],
                 storage_location: "Optional[PathLike[str]]" = None, filename: "Optional[PathLike[str]]" = None) -> None:
        super().__init__(estimator, encoder, storage_location, filename)
        self.vectorizer = vectorizer

    def encode_x(self, 
                 instances: Iterable[Instance[KT, Any, np.ndarray, Any]]) -> np.ndarray:
        def encode_or_vectorize():
            for ins in instances:
                if ins.vector is not None:
                    yield ins.vector
                else:
                    ins.vector = self.vectorizer.transform([ins]) # type: ignore
                    yield ins.vector
        x_data = list(encode_or_vectorize())
        x_vec = np.vstack(x_data)
        return x_vec

    def encode_xy(self, instances: Iterable[Instance[KT, Any, np.ndarray, Any]], 
                        labelings: Iterable[Iterable[LT]]):
        def yield_xy():
            for ins, lbl in zip(instances, labelings):
                if ins.vector is not None:
                    ins.vector = list(self.vectorizer.transform([ins]))[0] # type: ignore
                encoded_label = self.encoder.encode_safe(lbl)
                if encoded_label is not None:
                    yield ins.vector, encoded_label
        x_data, y_data = list_unzip(yield_xy())
        x_fm = np.vstack(x_data)
        y_lm = np.vstack(y_data)
        if y_lm.shape[1] == 1:
            y_lm = np.reshape(y_lm, (y_lm.shape[0],))
        return x_fm, y_lm

    def _vectorize(self, provider: InstanceProvider[IT, KT, Any, np.ndarray, Any]) -> None:
        without_vectors = provider.without_vector
        if without_vectors:
            keys = list(without_vectors)
            inss = [provider[key] for key in keys]
            vec_list = list(self.vectorizer.transform(inss))
            provider.bulk_add_vectors(keys, vec_list)

    def predict_proba_provider(self, provider: InstanceProvider[IT, KT, Any, np.ndarray, Any], batch_size: int = 200) -> Sequence[Tuple[KT, FrozenSet[Tuple[LT, float]]]]:
        self._vectorize(provider)
        return super().predict_proba_provider(provider, batch_size)
    
    def predict_proba_provider_raw(self, provider: InstanceProvider[IT, KT, Any, np.ndarray, Any], batch_size: int = 200) -> Iterator[Tuple[Sequence[KT], np.ndarray]]:
        self._vectorize(provider)
        return super().predict_proba_provider_raw(provider, batch_size)

    def predict_provider(self, provider: InstanceProvider[IT, KT, Any, np.ndarray, Any], batch_size: int = 200) -> Sequence[Tuple[KT, FrozenSet[LT]]]:
        self._vectorize(provider)
        return super().predict_provider(provider, batch_size)

    @classmethod
    def from_skvector(cls, 
                       model: SkLearnVectorClassifier[IT, KT, LT], 
                       vectorizer: BaseVectorizer[IT]) -> AutoVectorizerClassifier[IT, KT, LT]:
        result = cls(model.innermodel, vectorizer, model.encoder)
        return result
