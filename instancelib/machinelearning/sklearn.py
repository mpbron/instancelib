# Copyright (C) 2021 The InstanceLib Authors. All Rights Reserved.

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 3 of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
from __future__ import annotations

import functools
import itertools
import logging
import operator
from abc import ABC, abstractmethod
from os import PathLike
from typing import (Any, FrozenSet, Generic, Iterable, Iterator, Optional,
                    Sequence, Tuple, TypeVar, Union, List)

import numpy as np
from sklearn.pipeline import Pipeline  # type: ignore

from sklearn.preprocessing import LabelEncoder as SKLabelEncoder # type: ignore
from sklearn.preprocessing import MultiLabelBinarizer # type: ignore
from sklearn.base import ClassifierMixin, TransformerMixin  # type: ignore

from ..environment import Environment
from ..environment.base import Environment
from ..instances import Instance, InstanceProvider
from ..labels.encoder import (DictionaryEncoder, LabelEncoder, MultilabelDictionaryEncoder, SklearnLabelEncoder,
                              SklearnMultiLabelEncoder)
from ..typehints.typevars import DT, KT, LT, VT
from ..utils import SaveableInnerModel
from ..utils.chunks import divide_iterable_in_lists

from .base import AbstractClassifier

LOGGER = logging.getLogger(__name__)

IT = TypeVar("IT", bound="Instance[Any, Any, np.ndarray, Any]", covariant=True)

class SkLearnClassifier(SaveableInnerModel,
                        AbstractClassifier[IT, KT, DT, VT, Any, LT, np.ndarray, np.ndarray], 
                        ABC, Generic[IT, KT, DT, VT, LT]):
    _name = "Sklearn"
    
    def __init__(
            self,
            estimator: Union[ClassifierMixin, Pipeline], 
            encoder: LabelEncoder[LT, np.ndarray, np.ndarray, np.ndarray],
            storage_location: "Optional[PathLike[str]]"=None, 
            filename: "Optional[PathLike[str]]"=None
            ) -> None:
        SaveableInnerModel.__init__(self, estimator, storage_location, filename)
        self.encoder = encoder 
        self._fitted = False

    def set_target_labels(self, labels: Iterable[LT]) -> None:
        self.encoder.initialize(labels)

    @SaveableInnerModel.load_model_fallback
    def _fit(self, x_data: np.ndarray, y_data: np.ndarray):
        assert x_data.shape[0] == y_data.shape[0]
        self.innermodel.fit(x_data, y_data) # type: ignore
        LOGGER.info("[%s] Fitted the model", self.name)
        self._fitted = True

    @SaveableInnerModel.load_model_fallback
    def _predict_proba(self, x_data: np.ndarray) -> np.ndarray:
        assert self.innermodel is not None
        return self.innermodel.predict_proba(x_data) 

    @SaveableInnerModel.load_model_fallback
    def _predict(self, x_data: np.ndarray) -> np.ndarray:
        assert self.innermodel is not None
        return self.innermodel.predict(x_data)

    @abstractmethod
    def encode_x(self, 
                 instances: Iterable[Instance[KT, DT, VT, Any]]) -> np.ndarray:
        raise NotImplementedError

    def encode_y(self, labelings: Sequence[Iterable[LT]]) -> np.ndarray:
        y_data = self.encoder.encode_batch(labelings)
        return y_data
        
    @abstractmethod
    def encode_xy(self, instances: Iterable[Instance[KT, DT, VT, Any]], 
                        labelings: Iterable[Iterable[LT]]) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def fit_instances(self, instances: Iterable[Instance[KT, DT, VT, Any]], labels: Iterable[Iterable[LT]]) -> None:
        x_train_vec, y_train_vec = self.encode_xy(instances, labels)
        self._fit(x_train_vec, y_train_vec)

    def _pred_ins_batch(self, batch: Iterable[Instance[KT, DT, VT, Any]]) -> Sequence[Tuple[KT, FrozenSet[LT]]]:
        x_keys = [ins.identifier for ins in batch]
        x_vec = self.encode_x(batch)
        y_pred = self._predict(x_vec)
        labels = self.encoder.decode_matrix(y_pred)
        zipped = list(zip(x_keys, labels))
        return zipped

    def _pred_proba_raw_ins_batch(self, batch: Iterable[Instance[KT, DT, VT, Any]]) -> Tuple[Sequence[KT], np.ndarray]:
        x_keys = [ins.identifier for ins in batch]
        x_vec  = self.encode_x(batch)
        y_pred = self._predict_proba(x_vec)
        return x_keys, y_pred
        
    def _pred_proba_ins_batch(self, batch: Iterable[Instance[KT, DT, VT, Any]]) -> Sequence[Tuple[KT, FrozenSet[Tuple[LT, float]]]]:
        x_keys = [ins.identifier for ins in batch]
        x_vec = self.encode_x(batch)
        y_pred = self._predict_proba(x_vec)
        labels = self.encoder.decode_proba_matrix(y_pred)
        zipped = list(zip(x_keys, labels))
        return zipped

    def predict_proba_instances_raw(self, 
                                    instances: Iterable[Instance[KT, DT, VT, Any]],
                                    batch_size: int = 200
                                    ) -> Iterator[Tuple[Sequence[KT], np.ndarray]]:
        batches = divide_iterable_in_lists(instances, batch_size)
        processed = map(self._pred_proba_raw_ins_batch, batches)
        yield from processed

    def predict_proba_instances(self, 
                                instances: Iterable[Instance[KT, DT, VT, Any]],
                                batch_size: int = 200
                                ) -> Sequence[Tuple[KT, FrozenSet[Tuple[LT, float]]]]:
        
        batches = divide_iterable_in_lists(instances, batch_size)
        processed = map(self._pred_proba_ins_batch, batches)
        combined: Sequence[Tuple[KT, FrozenSet[Tuple[LT, float]]]] = functools.reduce(
            operator.concat, processed) # type: ignore
        return combined

    def predict_instances(self, 
                          instances: Iterable[Instance[KT, DT, VT, Any]],
                          batch_size: int = 200) -> Sequence[Tuple[KT, FrozenSet[LT]]]:        
        batches = divide_iterable_in_lists(instances, batch_size)
        results = map(self._pred_ins_batch, batches)
        concatenated: Sequence[Tuple[KT, FrozenSet[LT]]] = functools.reduce(
            lambda a,b: operator.concat(a,b), results) # type: ignore
        return concatenated

    def _decode_proba_matrix(self, 
                             keys: Sequence[KT], 
                             y_matrix: np.ndarray) -> Sequence[Tuple[KT, FrozenSet[Tuple[LT, float]]]]:
        y_labels = self.encoder.decode_proba_matrix(y_matrix)
        zipped = list(zip(keys, y_labels)) 
        return zipped

    def predict_proba_provider(self, 
                               provider: InstanceProvider[IT, KT, DT, VT, Any], 
                               batch_size: int = 200) -> Sequence[Tuple[KT, FrozenSet[Tuple[LT, float]]]]:
        preds = self.predict_proba_provider_raw(provider, batch_size)
        decoded_probas = itertools.starmap(self._decode_proba_matrix, preds)
        chained = list(itertools.chain.from_iterable(decoded_probas))
        return chained

    @property
    def name(self) -> str:
        if self.innermodel is not None:
            return f"{self._name} :: {self.innermodel.__class__}"
        return f"{self._name} :: No Innermodel Present"

    @property
    def fitted(self) -> bool:
        return self._fitted

    @classmethod
    def build_from_model(cls,
                         estimator: Union[ClassifierMixin, Pipeline],
                         classes: Optional[Sequence[LT]] = None,
                         storage_location: "Optional[PathLike[str]]"=None, 
                         filename: "Optional[PathLike[str]]"=None
                         ) -> SkLearnClassifier[IT, KT, DT, VT, LT]:
        """Construct a Sklearn Data model from a fitted Sklearn model.
        The estimator is a classifier for a binary or multiclass classification problem.

        Parameters
        ----------
        estimator : ClassifierMixin
            The Sklearn Classifier (e.g., :class:`sklearn.naive_bayes.MultinomialNB`).
            The field `classes_` is used to decode the label predictions.
        classes : Optional[Sequence[LT]]
            The position of each label, optional
        storage_location : Optional[PathLike[str]], optional
            If you want to save the model, you can specify the storage folder, by default None
        filename : Optional[PathLike[str]], optional
            If the model has a specific filename, you can specify it here, by default None

        Returns
        -------
        SkLearnClassifier[IT, KT, DT, LT]
            The model
        """
        if classes is None:
            labels: List[LT] = estimator.classes_.tolist() # type: ignore
            il_encoder = DictionaryEncoder[LT].from_list(labels)
        else:
            il_encoder = DictionaryEncoder[LT].from_list(classes)
        return cls(estimator, il_encoder, storage_location, filename)

    @classmethod
    def build_from_model_multilabel(cls,
                         estimator: Union[ClassifierMixin, Pipeline],
                         classes: Optional[Sequence[LT]] = None,
                         storage_location: "Optional[PathLike[str]]"=None, 
                         filename: "Optional[PathLike[str]]"=None
                         ) -> SkLearnClassifier[IT, KT, DT, VT, LT]:
        """Construct a Sklearn Data model from a fitted Sklearn model.
        The estimator is a classifier for a multilabel classification problem.

        Parameters
        ----------
        estimator : ClassifierMixin
            The Sklearn Classifier (e.g., :class:`sklearn.naive_bayes.MultinomialNB`).
            The field `classes_` is used to decode the label predictions.
        classes : Optional[Sequence[LT]]
            The position of each label, optional
        storage_location : Optional[PathLike[str]], optional
            If you want to save the model, you can specify the storage folder, by default None
        filename : Optional[PathLike[str]], optional
            If the model has a specific filename, you can specify it here, by default None

        Returns
        -------
        SkLearnClassifier[IT, KT, DT, LT]
            The model
        """
        if classes is None:
            labels: List[LT] = estimator.classes_.tolist() # type: ignore
            il_encoder = MultilabelDictionaryEncoder[LT].from_list(labels)
        else:
            il_encoder = MultilabelDictionaryEncoder[LT].from_list(classes)
        return cls(estimator, il_encoder, storage_location, filename)
        

    @classmethod
    def build(cls,
                 estimator: Union[ClassifierMixin, Pipeline],
                 env: Environment[IT, KT, DT, VT, Any, LT],
                 storage_location: "Optional[PathLike[str]]"=None, 
                 filename: "Optional[PathLike[str]]"=None
                 ) -> SkLearnClassifier[IT, KT, DT, VT, LT]:
        """Construct a Sklearn Data model from an :class:`~instancelib.Environment`.
        The estimator is a classifier for a binary or multiclass classification problem.

        Parameters
        ----------
        estimator : ClassifierMixin
            The Sklearn Classifier (e.g., :class:`sklearn.naive_bayes.MultinomialNB`)
        env : Environment[IT, KT, Any, np.ndarray, Any, LT]
            The environment that will be used to gather the labels from
        storage_location : Optional[PathLike[str]], optional
            If you want to save the model, you can specify the storage folder, by default None
        filename : Optional[PathLike[str]], optional
            If the model has a specific filename, you can specify it here, by default None

        Returns
        -------
        SkLearnClassifier[IT, KT, DT, LT]
            The model
        """
        sklearn_encoder: TransformerMixin = SKLabelEncoder()
        il_encoder = SklearnLabelEncoder(sklearn_encoder, env.labels.labelset)    
        return cls(estimator, il_encoder, storage_location, filename)

    @classmethod
    def build_multilabel(cls,
                 estimator: Union[ClassifierMixin, Pipeline],
                 env: Environment[IT, KT, Any, np.ndarray, Any, LT],
                 storage_location: "Optional[PathLike[str]]"=None, 
                 filename: "Optional[PathLike[str]]"=None
                 ) -> SkLearnClassifier[IT, KT, DT, VT, LT]:
        """Construct a Sklearn Data model from an :class:`~instancelib.Environment`.
        The estimator is a classifier for a binary or multiclass classification problem.

        Parameters
        ----------
        estimator : ClassifierMixin
             The scikit-learn API Classifier capable of Multilabel Classification
        env : Environment[IT, KT, Any, np.ndarray, Any, LT]
            The environment that will be used to gather the labels from
        storage_location : Optional[PathLike[str]], optional
            If you want to save the model, you can specify the storage folder, by default None
        filename : Optional[PathLike[str]], optional
            If the model has a specific filename, you can specify it here, by default None

        Returns
        -------
        SkLearnClassifier[IT, KT, DT, VT, LT]:
            The model
        """        
        sklearn_encoder: TransformerMixin = MultiLabelBinarizer()
        il_encoder = SklearnMultiLabelEncoder(sklearn_encoder, env.labels.labelset)    
        return cls(estimator, il_encoder, storage_location, filename)
