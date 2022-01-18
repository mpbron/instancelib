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

import itertools
import logging
from os import PathLike
from typing import (Any, Callable, FrozenSet, Generic, Iterable, Iterator,
                    Optional, Sequence, Tuple, TypeVar, Union)

import numpy as np  # type: ignore

import sklearn as sk  # type: ignore
from sklearn.base import ClassifierMixin, TransformerMixin
from sklearn.pipeline import Pipeline

from ..instances import Instance, InstanceProvider
from ..labels.base import LabelProvider
from ..labels.encoder import LabelEncoder  # type: ignore
from ..typehints.typevars import DT, KT, LT
from ..utils.func import list_unzip, zip_chain
from .sklearn import SkLearnClassifier

LOGGER = logging.getLogger(__name__)

IT = TypeVar("IT", bound="Instance[Any, Any, np.ndarray, Any]", covariant=True)

class SkLearnDataClassifier(SkLearnClassifier[IT, KT, DT, Any, LT], 
                            Generic[IT, KT, DT, LT]):
    _name = "DataClassifier"

   
    def encode_x(self, 
                 instances: Iterable[Instance[KT, DT, Any, Any]]) -> np.ndarray:
        x_data = [instance.data for instance in instances]
        x_vec = np.array(x_data) # type: ignore
        return x_vec

    def encode_xy(self, instances: Iterable[Instance[KT, DT, Any, Any]], 
                        labelings: Iterable[Iterable[LT]]):
        def yield_xy():
            for ins, lbl in zip(instances, labelings):
                encoded_label = self.encoder.encode_safe(lbl)
                if encoded_label is not None:
                    yield ins.data, encoded_label
        x_data, y_data = list_unzip(yield_xy())
        x_fm = np.array(x_data) # type: ignore
        y_lm = np.vstack(y_data)
        if y_lm.shape[1] == 1:
            y_lm = np.reshape(y_lm, (y_lm.shape[0],))
        return x_fm, y_lm

    def _get_preds(self, tuples: Sequence[Tuple[KT, DT]]) -> Tuple[Sequence[KT], Sequence[FrozenSet[LT]]]:
        """Predict the labels for the current (key, data) tuples

        Parameters
        ----------
        tuples : Sequence[Tuple[KT, DT]]
            The tuples that we want the predictions from

        Returns
        -------
        Tuple[Sequence[KT], Sequence[FrozenSet[LT]]]
            A list of keys and the predictions belonging to it
        """        
        keys, data = list_unzip(tuples)
        data_vec: np.ndarray = np.array(data) # type: ignore
        pred_vec: np.ndarray = self._predict(data_vec)
        labels = self.encoder.decode_matrix(pred_vec)
        return keys, labels


    def _get_probas(self, tuples: Sequence[Tuple[KT, DT]]) -> Tuple[Sequence[KT], np.ndarray]:
        """Calculate the probability matrix for the current (key, data) tuples
        
        Parameters
        ----------
        tuples : Sequence[Tuple[KT, DT]]
            The tuples that we want the predictions from
        
        Returns
        -------
        Tuple[Sequence[KT], np.ndarray]
            A list of keys and the probability predictions belonging to it
        """
        keys, data = list_unzip(tuples)
        data_vec: np.ndarray = np.array(data) # type: ignore
        prob_vec: np.ndarray = self._predict_proba(data_vec)  # type: ignore
        return keys, prob_vec

    def predict_proba_provider_raw(self, 
                                   provider: InstanceProvider[IT, KT, DT, Any, Any],
                                   batch_size: int = 200) -> Iterator[Tuple[Sequence[KT], np.ndarray]]:
        tuples = provider.data_chunker(batch_size)
        preds = map(self._get_probas, tuples)
        yield from preds

    def predict_provider(self, 
                         provider: InstanceProvider[IT, KT, DT, Any, Any], 
                         batch_size: int = 200) -> Sequence[Tuple[KT, FrozenSet[LT]]]:
        tuples = provider.data_chunker(batch_size)
        preds = map(self._get_preds, tuples)
        results = list(zip_chain(preds))
        return results
        
    def fit_provider(self, 
                     provider: InstanceProvider[IT, KT, DT, Any, Any], 
                     labels: LabelProvider[KT, LT], 
                     batch_size: int = 200) -> None:
        LOGGER.info("[%s] Start with the fit procedure", self.name)
        keys, datas = list_unzip(itertools.chain.from_iterable(provider.data_chunker(batch_size)))
        LOGGER.info("[%s] Gathered all data", self.name)
        labelings = list(map(labels.get_labels, keys))
        LOGGER.info("[%s] Gathered all labels", self.name)
        self._fit_data(datas, labelings)
        LOGGER.info("[%s] Fitted the classifier", self.name)
        

    def _fit_data(self, x_data: Sequence[DT], labels: Sequence[FrozenSet[LT]]):
        x_dat, y_mat = self._filter_x_only_encoded_y(x_data, labels)
        x_mat = np.array(x_dat) # type: ignore
        self._fit(x_mat, y_mat)

    def fit_instances(self, instances: Iterable[Instance[KT, DT, Any, Any]], labels: Iterable[Iterable[LT]]) -> None:
        datas = [ins.data for ins in instances]
        labelsets = [frozenset(labeling) for labeling in labels]
        self._fit_data(datas, labelsets)


  
class SeparateDataEncoderClassifier(SkLearnDataClassifier[IT, KT, DT, LT], Generic[IT, KT, DT, LT]):
    input_encoder: Callable[[Sequence[DT]], np.ndarray]
    
    def __init__(self, 
                 estimator: Union[ClassifierMixin, Pipeline], 
                 encoder: LabelEncoder[LT, np.ndarray, np.ndarray, np.ndarray], 
                 storage_location: "Optional[PathLike[str]]" = None, 
                 filename: "Optional[PathLike[str]]" = None, 
                 input_encoder: Callable[[Sequence[DT]], np.ndarray] = np.array) -> None:
        super().__init__(estimator, encoder, storage_location=storage_location, filename=filename)
        self.input_encoder = input_encoder

    def encode_x(self, instances: Iterable[Instance[KT, DT, Any, Any]]) -> np.ndarray:
        dats = [instance.data for instance in instances]
        encoded = self.input_encoder(dats)
        return encoded

    def encode_xy(self, instances: Iterable[Instance[KT, DT, Any, Any]], 
                        labelings: Iterable[Iterable[LT]]):
        def yield_xy():
            for ins, lbl in zip(instances, labelings):
                encoded_label = self.encoder.encode_safe(lbl)
                if encoded_label is not None:
                    yield ins.data, encoded_label
        x_data, y_data = list_unzip(yield_xy())
        x_fm = self.input_encoder(x_data)
        y_lm = np.vstack(y_data)
        if y_lm.shape[1] == 1:
            y_lm = np.reshape(y_lm, (y_lm.shape[0],))
        return x_fm, y_lm

    def _get_preds(self, tuples: Sequence[Tuple[KT, DT]]) -> Tuple[Sequence[KT], Sequence[FrozenSet[LT]]]:
        keys, data = list_unzip(tuples)
        data_vec: np.ndarray = self.input_encoder(data)
        pred_vec: np.ndarray = self._predict(data_vec)
        labels = self.encoder.decode_matrix(pred_vec)
        return keys, labels

    def _get_probas(self, tuples: Sequence[Tuple[KT, DT]]) -> Tuple[Sequence[KT], np.ndarray]:
        keys, data = list_unzip(tuples)
        data_vec: np.ndarray = self.input_encoder(data)
        prob_vec: np.ndarray = self._predict_proba(data_vec)  # type: ignore
        return keys, prob_vec

    def _fit_data(self, x_data: Sequence[DT], labels: Sequence[FrozenSet[LT]]):
        x_dat, y_mat = self._filter_x_only_encoded_y(x_data, labels)
        x_mat = self.input_encoder(x_dat)
        self._fit(x_mat, y_mat)

    
