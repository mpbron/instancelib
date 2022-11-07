from __future__ import annotations

import itertools
import math

from typing import (
    Any,
    Callable,
    FrozenSet,
    Generic,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import numpy.typing as npt
from tqdm.auto import tqdm

from ..instances import Instance
from ..instances.base import InstanceProvider
from ..labels.base import LabelProvider
from ..labels.encoder import LabelEncoder, MultilabelDictionaryEncoder
from ..typehints import DT, KT, LMT, LT, LVT, PMT, RT, VT
from ..utils.chunks import divide_iterable_in_lists
from ..utils.func import invert_mapping, list_unzip, seq_or_map_to_map
from .base import AbstractClassifier

IT = TypeVar("IT", bound="Instance[Any, Any, Any, Any]")


def to_int(
    f: Callable[..., npt.NDArray[Any]]
) -> Callable[..., npt.NDArray[np.int64]]:
    def wrapper(*args, **kwargs):
        result = f(*args, **kwargs)
        int_result = result.astype(np.int64)
        return int_result

    return wrapper


def numpy_ova_threshold(
    threshold: float,
) -> Callable[[npt.NDArray[Any]], npt.NDArray[np.bool8]]:
    def mat_function(mat: npt.NDArray[Any]) -> npt.NDArray[np.bool8]:
        binary = mat > threshold
        return binary

    return mat_function


def numpy_mc_threshold(mat: npt.NDArray[Any]) -> npt.NDArray[np.bool8]:
    max_index: Sequence[int] = np.argmax(mat, axis=1).tolist()
    return_mat = np.zeros_like(mat).astype(np.bool8)
    for row_idx, col_idx in enumerate(max_index):
        return_mat[row_idx, col_idx] = True
    return return_mat


def data_chunker(
    instances: Iterable[Instance[KT, DT, Any, Any]], batch_size: int = 200
) -> Iterator[Sequence[Tuple[KT, DT]]]:
    chunks = divide_iterable_in_lists(instances, batch_size)
    for chunk in chunks:
        yield [(ins.identifier, ins.data) for ins in chunk]


class DataWrapper(
    AbstractClassifier[IT, KT, DT, VT, RT, LT, LMT, PMT],
    Generic[IT, KT, DT, VT, RT, LT, LVT, LMT, PMT],
):
    proba_vec_function: Callable[[Sequence[DT]], PMT]

    def __init__(
        self,
        proba_function: Callable[[Sequence[DT]], PMT],
        threshold_func: Callable[[PMT], LMT],
        encoder: LabelEncoder[LT, LVT, LMT, PMT],
    ) -> None:
        self.proba_vec_function = proba_function
        self.threshold_function = threshold_func
        self.encoder = encoder

    @property
    def fitted(self) -> bool:
        return True

    def fit_instances(
        self,
        instances: Iterable[Instance[KT, DT, VT, RT]],
        labels: Iterable[Iterable[LT]],
    ) -> None:
        pass

    def fit_provider(
        self,
        provider: InstanceProvider[IT, KT, DT, VT, RT],
        labels: LabelProvider[KT, LT],
        batch_size: int = 200,
    ) -> None:
        pass

    def _get_probas(
        self, tuples: Sequence[Tuple[KT, DT]]
    ) -> Tuple[Sequence[KT], PMT]:
        """Calculate the probability matrix for the current (key, data) tuples

        Parameters
        ----------
        tuples : Sequence[Tuple[KT, DT]]
            The tuples that we want the predictions from

        Returns
        -------
        Tuple[Sequence[KT], npt.NDArray[Any]]
            A list of keys and the probability predictions belonging to it
        """
        keys, data = list_unzip(tuples)
        y_pred = self.proba_vec_function(data)
        return keys, y_pred

    def _proba_iterator(
        self, datas: Sequence[Tuple[KT, DT]]
    ) -> Sequence[Tuple[KT, FrozenSet[Tuple[LT, float]]]]:
        keys, y_pred = self._get_probas(datas)
        labels = self.encoder.decode_proba_matrix(y_pred)
        return list(zip(keys, labels))

    def _pred_iterator(
        self, datas: Sequence[Tuple[KT, DT]]
    ) -> Sequence[Tuple[KT, FrozenSet[LT]]]:
        return self._decode_proba_matrix_pred(*self._get_probas(datas))

    def predict_proba_instances(
        self,
        instances: Iterable[Instance[KT, DT, VT, RT]],
        batch_size: int = 200,
    ) -> Sequence[Tuple[KT, FrozenSet[Tuple[LT, float]]]]:
        decoded_probas = map(
            self._proba_iterator, data_chunker(instances, 200)
        )
        chained = list(itertools.chain.from_iterable(decoded_probas))
        return chained

    def _decode_proba_matrix(
        self, keys: Sequence[KT], y_matrix: PMT
    ) -> Sequence[Tuple[KT, FrozenSet[Tuple[LT, float]]]]:
        y_labels = self.encoder.decode_proba_matrix(y_matrix)
        zipped = list(zip(keys, y_labels))
        return zipped

    def _decode_proba_matrix_pred(
        self, keys: Sequence[KT], y_matrix: PMT
    ) -> Sequence[Tuple[KT, FrozenSet[LT]]]:
        thresholded = self.threshold_function(y_matrix)
        y_labels = self.encoder.decode_matrix(thresholded)
        zipped = list(zip(keys, y_labels))
        return zipped

    def predict_instances(
        self,
        instances: Iterable[Instance[KT, DT, VT, RT]],
        batch_size: int = 200,
    ) -> Sequence[Tuple[KT, FrozenSet[LT]]]:
        decoded_probas = map(self._pred_iterator, data_chunker(instances, 200))
        chained = list(itertools.chain.from_iterable(decoded_probas))
        return chained

    def predict_provider(
        self,
        provider: InstanceProvider[IT, KT, DT, VT, RT],
        batch_size: int = 200,
    ) -> Sequence[Tuple[KT, FrozenSet[LT]]]:
        decoded_probas = map(
            self._pred_iterator, provider.data_chunker(batch_size)
        )
        chained = list(itertools.chain.from_iterable(decoded_probas))
        return chained

    def predict_proba_provider_raw(
        self,
        provider: InstanceProvider[IT, KT, DT, Any, Any],
        batch_size: int = 200,
    ) -> Iterator[Tuple[Sequence[KT], PMT]]:
        tuples = provider.data_chunker(batch_size)
        total_it = math.ceil(len(provider) / batch_size)
        preds = map(
            self._get_probas, tqdm(tuples, total=total_it, leave=False)
        )
        yield from preds

    def predict_proba_instances_raw(
        self,
        instances: Iterable[Instance[KT, DT, VT, RT]],
        batch_size: int = 200,
    ) -> Iterator[Tuple[Sequence[KT], PMT]]:
        tuples = data_chunker(instances, batch_size)
        preds = map(self._get_probas, tqdm(tuples, leave=False))
        yield from preds

    def predict_proba_provider(
        self,
        provider: InstanceProvider[IT, KT, DT, VT, Any],
        batch_size: int = 200,
    ) -> Sequence[Tuple[KT, FrozenSet[Tuple[LT, float]]]]:
        preds = self.predict_proba_provider_raw(provider, batch_size)
        decoded_probas = itertools.starmap(self._decode_proba_matrix, preds)
        chained = list(itertools.chain.from_iterable(decoded_probas))
        return chained

    def get_label_column_index(self, label: LT) -> int:
        return self.encoder.get_label_column_index(label)

    def set_target_labels(self, labels: Iterable[LT]) -> None:
        pass

    @classmethod
    def from_vectorized_function(
        cls,
        function: Callable[[Sequence[DT]], PMT],
        threshold_func: Callable[[PMT], LMT],
        labels: Union[Sequence[LT], Mapping[int, LT]],
    ) -> DataWrapper[IT, KT, DT, VT, RT, LT, LVT, LMT, PMT]:
        inverted_mapping = invert_mapping(seq_or_map_to_map(labels))
        encoder = MultilabelDictionaryEncoder(inverted_mapping)
        model = cls(function, threshold_func, encoder)
        return model


class NumpyMultiClass(
    DataWrapper[
        IT,
        KT,
        DT,
        Any,
        Any,
        LT,
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.float64],
    ],
    Generic[IT, KT, DT, LT],
):
    @classmethod
    def build(
        cls,
        function: Callable[[Sequence[DT]], npt.NDArray[np.float64]],
        labels: Union[Sequence[LT], Mapping[int, LT]],
    ) -> DataWrapper[
        IT,
        KT,
        DT,
        Any,
        Any,
        LT,
        npt.NDArray[Any],
        npt.NDArray[Any],
        npt.NDArray[Any],
    ]:
        model = cls.from_vectorized_function(
            function, to_int(numpy_mc_threshold), labels
        )
        return model


class NumpyMultiLabel(
    NumpyMultiClass[IT, KT, DT, LT], Generic[IT, KT, DT, LT]
):
    @classmethod
    def build(
        cls,
        function: Callable[[Sequence[DT]], npt.NDArray[Any]],
        labels: Union[Sequence[LT], Mapping[int, LT]],
    ) -> DataWrapper[
        IT,
        KT,
        DT,
        Any,
        Any,
        LT,
        npt.NDArray[Any],
        npt.NDArray[Any],
        npt.NDArray[Any],
    ]:
        model = cls.from_vectorized_function(
            function, to_int(numpy_ova_threshold(0.5)), labels
        )
        return model
