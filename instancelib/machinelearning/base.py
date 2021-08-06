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

from abc import ABC, abstractmethod
from typing import (FrozenSet, Generic, Iterable, Iterator,
                    Sequence, Tuple, TypeVar, Any, Union)

from ..labels import LabelProvider
from ..instances import Instance, InstanceProvider

from ..typehints import KT, VT, DT, RT, LT, LMT, PMT

IT = TypeVar("IT", bound="Instance[Any,Any,Any,Any]", covariant = True)

InstanceInput = Union[InstanceProvider[IT, KT, DT, VT, RT], Iterable[Instance[KT, DT, VT, RT]]]

class AbstractClassifier(ABC, Generic[IT, KT, DT, VT, RT, LT, LMT, PMT]):
    _name = "AbstractClassifier"

    @abstractmethod
    def set_target_labels(self, labels: Iterable[LT]) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict_instances(self, 
                          instances: Iterable[Instance[KT, DT, VT, RT]], 
                          batch_size: int = 200) -> Sequence[Tuple[KT, FrozenSet[LT]]]:
        raise NotImplementedError

    @abstractmethod
    def predict_provider(self, 
                         provider: InstanceProvider[IT, KT, DT, VT, RT],
                         batch_size: int = 200
                         ) -> Sequence[Tuple[KT, FrozenSet[LT]]]:
        raise NotImplementedError

    @abstractmethod
    def predict_proba_provider(self, 
                         provider: InstanceProvider[IT, KT, DT, VT, RT],
                         batch_size: int = 200
                         ) -> Sequence[Tuple[KT, FrozenSet[Tuple[LT, float]]]]:
        raise NotImplementedError

    @abstractmethod
    def predict_proba_provider_raw(self, 
                         provider: InstanceProvider[IT, KT, DT, VT, RT],
                         batch_size: int = 200
                         ) -> Iterator[Tuple[Sequence[KT], PMT]]:
        raise NotImplementedError

    @abstractmethod
    def predict_proba_instances(self, 
                                instances: Iterable[Instance[KT, DT, VT, RT]],
                                batch_size: int = 200
                                ) -> Sequence[Tuple[KT, FrozenSet[Tuple[LT, float]]]]:
        raise NotImplementedError

    @abstractmethod
    def predict_proba_instances_raw(self, 
                                    instances: Iterable[Instance[KT, DT, VT, RT]],
                                    batch_size: int = 200
                                    ) -> Iterator[Tuple[Sequence[KT], PMT]]:
        raise NotImplementedError


    @abstractmethod
    def fit_provider(self, 
                     provider: InstanceProvider[IT, KT, DT, VT, RT],
                     labels: LabelProvider[KT, LT], batch_size: int = 200) -> None:
        raise NotImplementedError

    @abstractmethod
    def fit_instances(self, instances: Iterable[Instance[KT, DT, VT, RT]], labels: Iterable[Iterable[LT]]) -> None:
        raise NotImplementedError

   
    @property
    def name(self) -> str:
        return self._name
        
    @property
    @abstractmethod
    def fitted(self) -> bool:
        pass

    def predict(self, instances: InstanceInput[IT, KT, DT, VT, RT], batch_size: int = 200) -> Sequence[Tuple[KT, FrozenSet[LT]]]:
        """Predict the labels on input instances.

        Parameters
        ----------
        instances : InstanceInput[IT, KT, DT, VT, RT]
            An :class:`InstanceProvider` or :class:`Iterable` of :class:`Instance` objects.
        batch_size : int, optional
            A batch size, by default 200

        Returns
        -------
        Sequence[Tuple[KT, FrozenSet[LT]]]
            A Tuple of Keys corresponding with their labels

        Raises
        ------
        ValueError
            If you supply incorrect formatted arguments
        """        
        if isinstance(instances, InstanceProvider):
            typed_provider: InstanceProvider[IT, KT, DT, VT, RT] = instances # type: ignore
            result = self.predict_provider(typed_provider, batch_size)
            return result
        result = self.predict_instances(instances, batch_size)
        return result
        


    def predict_proba(self, instances: InstanceInput[IT, KT, DT, VT, RT], batch_size: int = 200) -> Sequence[Tuple[KT, FrozenSet[Tuple[LT, float]]]]:
        """Predict the labels and corresponding probabilities on input instances.

        Parameters
        ----------
        instances : InstanceInput[IT, KT, DT, VT, RT]
            An :class:`InstanceProvider` or :class:`Iterable` of :class:`Instance` objects.
        batch_size : int, optional
            A batch size, by default 200
        Returns
        -------
        Sequence[Tuple[KT, FrozenSet[Tuple[LT, float]]]]
             Tuple of Keys corresponding with tuples of probabilities and the labels

        Raises
        ------
        ValueError
            If you supply incorrect formatted arguments
        """        
        if isinstance(instances, InstanceProvider):
            typed_provider: InstanceProvider[IT, KT, DT, VT, RT] = instances # type: ignore
            result = self.predict_proba_provider(typed_provider, batch_size)
            return result
        preds = self.predict_proba_instances(instances, batch_size)
        return preds
        

    def predict_proba_raw(self, instances: InstanceInput[IT, KT, DT, VT, RT], batch_size: int = 200) -> Iterator[Tuple[Sequence[KT], PMT]]:
        if isinstance(instances, InstanceProvider):
            typed_provider: InstanceProvider[IT, KT, DT, VT, RT] = instances # type: ignore
            result = self.predict_proba_provider_raw(typed_provider, batch_size)
            return result
        preds = self.predict_proba_instances_raw(instances, batch_size)
        return preds