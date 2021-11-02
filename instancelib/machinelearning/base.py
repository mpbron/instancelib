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
    """This class provides an interface that can be used to connect your model to 
    :class:`~instancelib.InstanceProvider`, :class:`~instancelib.LabelProvider`, and 
    :class:`~instancelib.Environment` objects.
   
    The main methods of this class are listed below:

    - :meth:`fit_provider`: Fit a classifier on training instances
    - :meth:`predict`: Predict the class labels for (unseen) instances
    - :meth:`predict_proba`: Predict the class labels and corresponding probabilities
    - :meth:`predict_proba_raw`: Predicht the class probabilities and return them in matrix form

    Examples
    --------

    Fit a classifier on train data:
    
    >>> model.fit_provider(train, env.labels)

    Predict the class labels for a list of instances:

    >>> model.predict([ins])
    [(20, frozenset({"Games"}))]

    Return the class labels and probabilities:

    >>> model.predict_proba(test)
    [(20, frozenset({("Games", 0.66), ("Bedrijfsnieuws", 0.22), ("Smartphones", 0.12)})), ... ]

    Return the raw prediction matrix:

    >>> preds = model.predict_proba_raw(test, batch_size=512)
    >>> next(preds)
    ([3, 4, 5, ...], array([[0.143, 0.622, 0.233],
                            [0.278, 0.546, 0.175],
                            [0.726, 0.126, 0.146], 
                            ...]))
    """  
    _name = "AbstractClassifier"

    @abstractmethod
    def get_label_column_index(self, label: LT) -> int:
        """Return the column in which the labels are stored 
        in the label and prediction matrices

        Parameters
        ----------
        label : LT
            The label

        Returns
        -------
        int
            The column index of the label
        """        
        raise NotImplementedError

    @abstractmethod
    def set_target_labels(self, labels: Iterable[LT]) -> None:
        """Set the target labels of the classifier

        Parameters
        ----------
        labels : Iterable[LT]
            The class labels that the classifier can predict
        """        
        raise NotImplementedError

    @abstractmethod
    def predict_instances(self, 
                          instances: Iterable[Instance[KT, DT, VT, RT]], 
                          batch_size: int = 200) -> Sequence[Tuple[KT, FrozenSet[LT]]]:
        """Predict the labels for a :term:`iterable` of instances

        Parameters
        ----------
        instances : Iterable[Instance[KT, DT, VT, RT]]
            The instances
        batch_size : int, optional
            The batch size, by default 200

        Returns
        -------
        Sequence[Tuple[KT, FrozenSet[LT]]]
            A sequence of (identifier, prediction) pairs
        """        
        raise NotImplementedError

    @abstractmethod
    def predict_provider(self, 
                         provider: InstanceProvider[IT, KT, DT, VT, RT],
                         batch_size: int = 200
                         ) -> Sequence[Tuple[KT, FrozenSet[LT]]]:
        """Predict the labels for all instances in 
        an :class:`InstanceProvider`.

        Parameters
        ----------
        instances : InstanceProvider[IT, KT, DT, VT, RT]
            The instanceprovider
        batch_size : int, optional
            The batch size, by default 200

        Returns
        -------
        Sequence[Tuple[KT, FrozenSet[LT]]]
            A sequence of (identifier, prediction) pairs
        """        
        raise NotImplementedError

    @abstractmethod
    def predict_proba_provider(self, 
                         provider: InstanceProvider[IT, KT, DT, VT, RT],
                         batch_size: int = 200
                         ) -> Sequence[Tuple[KT, FrozenSet[Tuple[LT, float]]]]:
        """Predict the labels for each instance in the provider and
        return the probability for each label.

        Parameters
        ----------
        provider : InstanceProvider[IT, KT, DT, VT, RT]
            The provider
        batch_size : int, optional
            The batch size, by default 200

        Returns
        -------
        Sequence[Tuple[KT, FrozenSet[Tuple[LT, float]]]]
            A sequence of tuples consisting of:

            - The instance identifier
            - The class labels and their probabilities
        """        
        raise NotImplementedError

    @abstractmethod
    def predict_proba_provider_raw(self, 
                         provider: InstanceProvider[IT, KT, DT, VT, RT],
                         batch_size: int = 200
                         ) -> Iterator[Tuple[Sequence[KT], PMT]]:
        """Generator function that predicts the labels 
        for each instance in the provider. 
        The generator lazy evaluates the prediction function
        on batches of instances and yields class probabilities 
        in matrix form.

        Parameters
        ----------
        provider
            The input InstanceProvider
        batch_size : int, optional
            The batch size in which instances are processed, by default 200
            This also influences the shape of the resulting 
            probability matrix.

        Yields
        -------
        Iterator[Tuple[Sequence[KT], PMT]]
            An iterator yielding tuples consisting of:
                
                - A sequence of keys that match the rows of
                the probability matrix
                - The Probability matrix with shape 
                ``(len(keys), batch_size)``
        """        
        raise NotImplementedError

    @abstractmethod
    def predict_proba_instances(self, 
                                instances: Iterable[Instance[KT, DT, VT, RT]],
                                batch_size: int = 200
                                ) -> Sequence[Tuple[KT, FrozenSet[Tuple[LT, float]]]]:
        """Predict the labels for each instance in the provider and
        return the probability for each label.

        Parameters
        ----------
        instances  
            Input instances
        batch_size : int, optional
            The batch size, by default 200

        Returns
        -------
        Sequence[Tuple[KT, FrozenSet[Tuple[LT, float]]]]
            A sequence of tuples consisting of:

            - The instance identifier
            - The class labels and their probabilities
        """        
        raise NotImplementedError

    @abstractmethod
    def predict_proba_instances_raw(self, 
                                    instances: Iterable[Instance[KT, DT, VT, RT]],
                                    batch_size: int = 200
                                    ) -> Iterator[Tuple[Sequence[KT], PMT]]:
        """Generator function that predicts the labels 
        for each instance.
        The generator lazy evaluates the prediction function
        on batches of instances and yields class probabilities 
        in matrix form.

        Parameters
        ----------
        instances
            Input instances
        batch_size : int, optional
            The batch size in which instances are processed, by default 200
            This also influences the shape of the resulting 
            probability matrix.

        Yields
        -------
        Tuple[Sequence[KT], PMT]
            An iterator yielding tuples consisting of:
                
                - A sequence of keys that match the rows of the probability matrix
                - The Probability matrix with shape ``(batch_size, n_labels)``
        """  
        raise NotImplementedError


    @abstractmethod
    def fit_provider(self, 
                     provider: InstanceProvider[IT, KT, DT, VT, RT],
                     labels: LabelProvider[KT, LT], batch_size: int = 200) -> None:
        """Fit the classifier with the instances found in the 
        :class:`InstanceProvider` based on the labels in the 
        :class:`LabelProvider`

        Parameters
        ----------
        provider : InstanceProvider[IT, KT, DT, VT, RT]
            The provider that contains the training data
        labels : LabelProvider[KT, LT]
            The provider that contains the labels of the training data
        batch_size : int, optional
            A batch size for the training process, by default 200
        """                
        raise NotImplementedError

    @abstractmethod
    def fit_instances(self, instances: Iterable[Instance[KT, DT, VT, RT]], labels: Iterable[Iterable[LT]]) -> None:
        """Fit the classifier with the instances and accompanied
        labels found in the arguments.

        Parameters
        ----------
        instances : Iterable[Instance[KT, DT, VT, RT]]
            The train data
        labels : Iterable[Iterable[LT]]
            The labels of the train data
        """        
        raise NotImplementedError

   
    @property
    def name(self) -> str:
        """The name of the classifier

        Returns
        -------
        str
            A name that can be used to identify the classifier
        """        
        return self._name
        
    @property
    @abstractmethod
    def fitted(self) -> bool:
        """Return true if the classifier has been fitted

        Returns
        -------
        bool
            True if the classifier has been fitted
        """        
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
        """Generator function that predicts the labels 
        for each instance.
        The generator lazy evaluates the prediction function
        on batches of instances and yields class probabilities 
        in matrix form.

        Parameters
        ----------
        instances
            Input instances
        batch_size : int, optional
            The batch size in which instances are processed, by default 200
            This also influences the shape of the resulting 
            probability matrix.

        Yields
        -------
        Tuple[Sequence[KT], PMT]
            An iterator yielding tuples consisting of:
                
                - A sequence of keys that match the rows of the probability matrix
                - The Probability matrix with shape ``(batch_size, n_labels)``
        """
        if isinstance(instances, InstanceProvider):
            typed_provider: InstanceProvider[IT, KT, DT, VT, RT] = instances # type: ignore
            result = self.predict_proba_provider_raw(typed_provider, batch_size)
            return result
        preds = self.predict_proba_instances_raw(instances, batch_size)
        return preds