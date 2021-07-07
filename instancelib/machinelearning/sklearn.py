from __future__ import annotations
from instancelib.machinelearning.featurematrix import FeatureMatrix

import itertools
import logging
from os import PathLike
from typing import (Any, FrozenSet, Generic, Iterable, Iterator, List,
                    Optional, Sequence, Tuple)

import numpy as np  # type: ignore
from instancelib.instances import Instance, InstanceProvider
from instancelib.labels.base import LabelProvider

from sklearn.base import ClassifierMixin, TransformerMixin  # type: ignore

from ..exceptions import NoVectorsException
from ..environment import AbstractEnvironment
from ..typehints.typevars import KT, LT
from ..utils import SaveableInnerModel
from ..utils.func import list_unzip, zip_chain
from .base import AbstractClassifier

LOGGER = logging.getLogger(__name__)

class SkLearnVectorClassifier(SaveableInnerModel, 
                              AbstractClassifier[
                                    Instance[KT, Any, np.ndarray, Any], 
                                    KT, Any, np.ndarray, Any, LT, 
                                    np.ndarray, np.ndarray
                              ], 
                              Generic[KT, LT]):
    _name = "Sklearn"

    def __init__(
            self,
            estimator: ClassifierMixin, 
            encoder: TransformerMixin,
            storage_location: "Optional[PathLike[str]]"=None, 
            filename: "Optional[PathLike[str]]"=None
            ) -> None:
        SaveableInnerModel.__init__(self, estimator, storage_location, filename)
        self.encoder = encoder 
        self._fitted = False
        self._target_labels: FrozenSet[LT] = frozenset()

    def __call__(self, target_labels: Iterable[LT]) -> SkLearnVectorClassifier[KT, LT]:
        self._target_labels = frozenset(target_labels)
        self.encoder.fit(list(self._target_labels)) # type: ignore
        return self

    def encode_labels(self, labels: Iterable[LT]) -> np.ndarray:
        return self.encoder.transform(list(set(labels))) # type: ignore

    def decode_vector(self, vector: np.ndarray) -> Sequence[FrozenSet[LT]]:
        labelings: Iterable[LT] = self.encoder.inverse_transform(vector).tolist() # type: ignore
        return [frozenset([labeling]) for labeling in labelings]

    def get_label_column_index(self, label: LT) -> int:
        label_list: List[LT] = self.encoder.classes_.tolist() # type: ignore
        return label_list.index(label)

    @SaveableInnerModel.load_model_fallback
    def _fit(self, x_data: np.ndarray, y_data: np.ndarray):
        assert x_data.shape[0] == y_data.shape[0]
        x_resampled, y_resampled = x_data, y_data
        LOGGER.info("[%s] Balanced / Resampled the data", self.name)
        self.innermodel.fit(x_resampled, y_resampled) # type: ignore
        LOGGER.info("[%s] Fitted the model", self.name)
        self._fitted = True

    def encode_xy(self, instances: Sequence[Instance[KT, Any, np.ndarray, Any]], 
                        labelings: Sequence[Iterable[LT]]):
        def yield_xy():
            for ins, lbl in zip(instances, labelings):
                if ins.vector is not None:
                    yield ins.vector, self.encode_labels(lbl)
        x_data, y_data = list_unzip(yield_xy())
        x_fm = np.vstack(x_data)
        y_lm = np.vstack(y_data)
        if y_lm.shape[1] == 1:
            y_lm = np.reshape(y_lm, (y_lm.shape[0],))
        return x_fm, y_lm

    def encode_x(self, instances: Sequence[Instance[KT, Any, np.ndarray, Any]]) -> np.ndarray:
        # TODO Maybe convert to staticmethod
        x_data = [
            instance.vector for instance in instances if instance.vector is not None]
        x_vec = np.vstack(x_data)
        return x_vec

    def encode_y(self, labelings: Sequence[Iterable[LT]]) -> np.ndarray:
        y_data = [self.encode_labels(labeling) for labeling in labelings]
        y_vec = np.vstack(y_data)
        if y_vec.shape[1] == 1:
            y_vec = np.reshape(y_vec, (y_vec.shape[0],))
        return y_vec

    @SaveableInnerModel.load_model_fallback
    def _predict_proba(self, x_data: np.ndarray) -> np.ndarray:
        assert self.innermodel is not None
        return self.innermodel.predict_proba(x_data) 

    @SaveableInnerModel.load_model_fallback
    def _predict(self, x_data: np.ndarray) -> np.ndarray:
        assert self.innermodel is not None
        return self.innermodel.predict(x_data)

    def predict_instances(self, instances: Sequence[Instance[KT, Any, np.ndarray, Any]]) -> Sequence[FrozenSet[LT]]:
        x_vec = self.encode_x(instances)
        y_pred = self._predict(x_vec)
        return self.decode_vector(y_pred)
        
    def predict_proba_instances_raw(self, 
                                    instances: Sequence[Instance[KT, Any, np.ndarray, Any]]
                                    ) -> Tuple[Sequence[KT], np.ndarray]:
        x_keys = [ins.identifier for ins in instances]
        x_vec  = self.encode_x(instances)
        y_pred = self._predict_proba(x_vec)
        return x_keys, y_pred

    def predict_proba_instances(self, 
                                instances: Sequence[Instance[KT, Any, np.ndarray, Any]]
                                ) -> Sequence[FrozenSet[Tuple[LT, float]]]:      
        x_vec = self.encode_x(instances)
        y_pred = self._predict_proba(x_vec).tolist()
        label_list: List[str] = self.encoder.classes_.tolist() # type: ignore
        y_labels: List[FrozenSet[Tuple[LT, float]]] = [
            frozenset(zip(label_list, y_vec)) # type: ignore
            for y_vec in y_pred
        ]
        return y_labels

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
        labels = self.decode_vector(pred_vec)
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

    def _decode_proba_matrix(self, keys: Sequence[KT], y_matrix: np.ndarray) -> Sequence[Tuple[KT, FrozenSet[Tuple[LT, float]]]]:
        y_pred = y_matrix.tolist()
        label_list: List[LT] = self.encoder.classes_.tolist() # type: ignore
        y_labels: List[FrozenSet[Tuple[LT, float]]] = [
            frozenset(zip(label_list, y_vec))
            for y_vec in y_pred
        ]
        assert len(keys) == len(y_labels)
        zipped = list(zip(keys, y_labels)) 
        return zipped

    def predict_proba_provider(self, 
                               provider: InstanceProvider[Instance[KT, Any, np.ndarray, Any], 
                               KT, Any, np.ndarray, Any], 
                               batch_size: int = 200) -> Sequence[Tuple[KT, FrozenSet[Tuple[LT, float]]]]:
        preds = self.predict_proba_provider_raw(provider, batch_size)
        decoded_probas = itertools.starmap(self._decode_proba_matrix, preds)
        chained = list(itertools.chain.from_iterable(decoded_probas))
        return chained

    def predict_proba_provider_raw(self, 
                                   provider: InstanceProvider[Instance[KT, Any, np.ndarray, Any], 
                                   KT, Any, np.ndarray, Any], batch_size: int) -> Iterator[Tuple[Sequence[KT], np.ndarray]]:
        matrices = FeatureMatrix[KT].generator_from_provider(provider, batch_size)
        preds = map(self._get_probas, matrices)
        yield from preds

    def predict_provider(self, 
                         provider: InstanceProvider[Instance[KT, Any, np.ndarray, Any], KT, Any, np.ndarray, Any], 
                         batch_size: int = 200,
                         ) -> Sequence[Tuple[KT, FrozenSet[LT]]]:
        matrices = FeatureMatrix[KT].generator_from_provider(provider, batch_size)
        preds = map(self._get_preds, matrices)
        results = list(zip_chain(preds))
        return results
        

        


    def fit_provider(self, 
                     provider: InstanceProvider[Instance[KT, Any, np.ndarray, Any], KT, Any, np.ndarray, Any], 
                     labels: LabelProvider[KT, LT], 
                     batch_size: int = 200) -> None:
        LOGGER.info("[%s] Start with the fit procedure", self.name)
        # Some sanity checks
               
        # Collect the feature matrix for the labeled subset
        key_vector_pairs = itertools.chain.from_iterable(provider.vector_chunker(batch_size))
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
        y_vec = self.encode_y(labels)
        self._fit(x_mat, y_vec)

    @property
    def name(self) -> str:
        if self.innermodel is not None:
            return f"{self._name} :: {self.innermodel.__class__}"
        return f"{self._name} :: No Innermodel Present"

    def fit_instances(self, instances: Sequence[Instance[KT, Any, np.ndarray, Any]], labels: Sequence[Iterable[LT]]) -> None:
        assert len(instances) == len(labels)
        x_train_vec, y_train_vec = self.encode_xy(instances, labels)
        self._fit(x_train_vec, y_train_vec)

    @property
    def fitted(self) -> bool:
        return self._fitted




class MultilabelSkLearnVectorClassifier(SkLearnVectorClassifier[KT, LT], Generic[KT, LT]):
    _name = "Multilabel Sklearn"
    def __call__(self, target_labels: Iterable[LT]) -> MultilabelSkLearnVectorClassifier[KT, LT]:
        self._target_labels = frozenset(target_labels)
        self.encoder.fit(list(map(lambda x: {x}, self._target_labels))) # type: ignore
        return self

    def encode_labels(self, labels: Iterable[LT]) -> np.ndarray:
        return self.encoder.transform([list(set(labels))]) # type: ignore

    def decode_vector(self, vector: np.ndarray) -> Sequence[FrozenSet[LT]]:
        labelings: Iterable[Iterable[LT]] = self.encoder.inverse_transform(vector) # type: ignore
        return [frozenset(labeling) for labeling in labelings]
