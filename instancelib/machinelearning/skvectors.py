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
from typing import (Any, FrozenSet, Generic, Iterable, Iterator, Sequence,
                    Tuple, TypeVar)

import numpy as np  # type: ignore

from ..exceptions import NoVectorsException
from ..instances import Instance, InstanceProvider
from ..labels.base import LabelProvider
from ..typehints.typevars import KT, LT
from ..utils.func import list_unzip, zip_chain
from .featurematrix import FeatureMatrix
from .sklearn import SkLearnClassifier

LOGGER = logging.getLogger(__name__)

IT = TypeVar("IT", bound="Instance[Any, Any, np.ndarray, Any]", covariant=True)

class SkLearnVectorClassifier(SkLearnClassifier[IT, KT, Any, np.ndarray, LT], 
                              Generic[IT, KT, LT]):
    _name = "SklearnVector"

    
    def encode_xy(self, instances: Iterable[Instance[KT, Any, np.ndarray, Any]], 
                        labelings: Iterable[Iterable[LT]]):
        def yield_xy():
            for ins, lbl in zip(instances, labelings):
                if ins.vector is not None:
                    yield ins.vector, self.encoder.encode(lbl)
        x_data, y_data = list_unzip(yield_xy())
        x_fm = np.vstack(x_data)
        y_lm = np.vstack(y_data)
        if y_lm.shape[1] == 1:
            y_lm = np.reshape(y_lm, (y_lm.shape[0],))
        return x_fm, y_lm

    def encode_x(self, 
                 instances: Iterable[Instance[KT, Any, np.ndarray, Any]]) -> np.ndarray:
        x_data = [ins.vector for ins in instances if ins.vector is not None]
        x_vec = np.vstack(x_data)
        return x_vec

    def _get_preds(self, matrix: FeatureMatrix[KT]) -> Tuple[Sequence[KT], Sequence[FrozenSet[LT]]]:
        """Predict the labels for the current feature matrix
        
        Parameters
        ----------
        matrix : FeatureMatrix[KT]
            The matrix for which we want to know the predictions
        
        Returns
        -------
        Tuple[Sequence[KT], Sequence[FrozenSet[LT]]]
            A list of keys and the predictions belonging to it
        """
        pred_vec: np.ndarray = self._predict(matrix.matrix)
        keys = matrix.indices
        labels = self.encoder.decode_matrix(pred_vec)
        return keys, labels


    def _get_probas(self, matrix: FeatureMatrix[KT]) -> Tuple[Sequence[KT], np.ndarray]:
        """Calculate the probability matrix for the current feature matrix
        
        Parameters
        ----------
        matrix : FeatureMatrix[KT]
            The matrix for which we want to know the predictions
        
        Returns
        -------
        Tuple[Sequence[KT], np.ndarray]
            A list of keys and the probability predictions belonging to it
        """
        prob_vec: np.ndarray = self._predict_proba(matrix.matrix)  # type: ignore
        keys = matrix.indices
        return keys, prob_vec

    def _decode_proba_matrix(self, 
                             keys: Sequence[KT], 
                             y_matrix: np.ndarray) -> Sequence[Tuple[KT, FrozenSet[Tuple[LT, float]]]]:
        y_labels = self.encoder.decode_proba_matrix(y_matrix)
        zipped = list(zip(keys, y_labels)) 
        return zipped

    def predict_proba_provider_raw(self, 
                                   provider: InstanceProvider[IT, KT, Any, np.ndarray, Any],
                                   batch_size: int = 200) -> Iterator[Tuple[Sequence[KT], np.ndarray]]:
        matrices = FeatureMatrix[KT].generator_from_provider(provider, batch_size)
        preds = map(self._get_probas, matrices)
        yield from preds

    def predict_provider(self, 
                         provider: InstanceProvider[IT, KT, Any, np.ndarray, Any], 
                         batch_size: int = 200) -> Sequence[Tuple[KT, FrozenSet[LT]]]:
        matrices = FeatureMatrix[KT].generator_from_provider(provider, batch_size)
        preds = map(self._get_preds, matrices)
        results = list(zip_chain(preds))
        return results
        
    def fit_provider(self, 
                     provider: InstanceProvider[IT, KT, Any, np.ndarray, Any], 
                     labels: LabelProvider[KT, LT], 
                     batch_size: int = 200) -> None:
        LOGGER.info("[%s] Start with the fit procedure", self.name)
        # Collect the feature matrix for the labeled subset
        key_vector_pairs = itertools.chain.from_iterable(
            provider.vector_chunker(batch_size))
        keys, vectors = list_unzip(key_vector_pairs)
        if not vectors:
            raise NoVectorsException("There are no vectors available for training the classifier")
        LOGGER.info("[%s] Gathered the feature matrix for all labeled documents", self.name)
        
        # Get all labels for documents in the labeled set
        labelings = list(map(labels.get_labels, keys))
        LOGGER.info("[%s] Gathered all labels", self.name)
        LOGGER.info("[%s] Start fitting the classifier", self.name)
        self._fit_vectors(vectors, labelings)
        LOGGER.info("[%s] Fitted the classifier", self.name)
        

    def _fit_vectors(self, x_data: Sequence[np.ndarray], labels: Sequence[FrozenSet[LT]]):
        x_mat = np.vstack(x_data)
        y_mat = self.encode_y(labels)
        self._fit(x_mat, y_mat)
    

