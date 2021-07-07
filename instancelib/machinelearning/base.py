from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (FrozenSet, Generic, Iterable, Iterator,
                    Sequence, Tuple, TypeVar, Any)

from ..labels import LabelProvider
from ..instances import Instance, InstanceProvider

from ..typehints import KT, VT, DT, RT, LT, LVT, PVT

IT = TypeVar("IT", bound="Instance[Any,Any,Any,Any]", covariant = True)

class AbstractClassifier(ABC, Generic[IT, KT, DT, VT, RT, LT, LVT, PVT]):
    _name = "AbstractClassifier"

    @abstractmethod
    def __call__(self, 
                 target_labels: Iterable[LT],
                 ) -> AbstractClassifier[IT, KT, DT, VT, RT, LT, LVT, PVT]:
        """Initialize the classifier by supplying the target labels
        
        Parameters
        ----------
        target_labels : Iterable[LT]
            A set with number of labels
        
        Returns
        -------
        AbstractClassifier
            [description]
        """        
        raise NotImplementedError

    @abstractmethod
    def encode_labels(self, labels: Iterable[LT]) -> LVT:
        pass

    @abstractmethod
    def predict_instances(self, instances: Sequence[Instance[KT, DT, VT, RT]]) -> Sequence[FrozenSet[LT]]:
        pass

    @abstractmethod
    def predict_provider(self, 
                         provider: InstanceProvider[Instance[KT, DT, VT, RT], KT, DT, VT, RT],
                         batch_size: int = 200
                         ) -> Sequence[Tuple[KT, FrozenSet[LT]]]:
        raise NotImplementedError

    @abstractmethod
    def predict_proba_provider(self, 
                         provider: InstanceProvider[Instance[KT, DT, VT, RT], KT, DT, VT, RT],
                         batch_size: int = 200
                         ) -> Sequence[Tuple[KT, FrozenSet[Tuple[LT, float]]]]:
        raise NotImplementedError

    @abstractmethod
    def predict_proba_provider_raw(self, 
                         provider: InstanceProvider[Instance[KT, DT, VT, RT], KT, DT, VT, RT],
                         batch_size: int = 200
                         ) -> Iterator[Tuple[Sequence[KT], PVT]]:
        raise NotImplementedError

    @abstractmethod
    def predict_proba_instances(self, 
                                instances: Sequence[Instance[KT, DT, VT, RT]]
                                ) -> Sequence[FrozenSet[Tuple[LT, float]]]:
        raise NotImplementedError

    @abstractmethod
    def predict_proba_instances_raw(self, 
                                    instances: Sequence[Instance[KT, DT, VT, RT]]
                                    ) -> Tuple[Sequence[KT], PVT]:
        raise NotImplementedError


    @abstractmethod
    def fit_provider(self, 
                     provider: InstanceProvider[Instance[KT, DT, VT, RT], KT, DT, VT, RT],
                     labels: LabelProvider[KT, LT], batch_size: int = 200) -> None:
        raise NotImplementedError

    @abstractmethod
    def fit_instances(self, instances: Sequence[Instance[KT, DT, VT, RT]], labels: Sequence[Iterable[LT]]) -> None:
        raise NotImplementedError

   
    @property
    def name(self) -> str:
        return self._name
        
    @property
    @abstractmethod
    def fitted(self) -> bool:
        pass

    @abstractmethod
    def get_label_column_index(self, label: LT) -> int:
        raise NotImplementedError
