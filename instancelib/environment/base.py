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

import random

from typing import Generic, Iterable, Sequence, Tuple, TypeVar, Any, Union
from abc import ABC, abstractmethod

from ..utils.func import union
from ..instances import InstanceProvider, Instance
from ..labels import LabelProvider

from ..typehints import KT, DT, VT, RT, LT

import warnings

InstanceType = TypeVar("InstanceType", bound="Instance[Any, Any, Any, Any]", covariant=True)

class Environment(ABC, Generic[InstanceType, KT, DT, VT, RT, LT]):
    """Environments provide an interface that enable you to access all data stored in the datasets.
    If there are labels stored in the environment, you can access these as well from here.

    There are two important properties in every :class:`Environment`:

    - :meth:`dataset`: Contains all Instances of the original dataset
    - :meth:`labels`: Contains an object that allows you to access labels easily

    Besides these properties, this object also provides methods to create new 
    :class:`~instancelib.InstanceProvider` objects that contain a subset of 
    the set of all instances stored in this environment.

    Examples
    --------

    Access the dataset:

    >>> dataset = env.dataset
    >>> instance = next(iter(dataset.values()))

    Access the labels:

    >>> labels = env.labels
    >>> ins_lbls = labels.get_labels(instance)

    Create a train-test split on the dataset (70 % train, 30 % test):

    >>> train, test = env.train_test_split(dataset, 0.70)
    """    
    @abstractmethod
    def create_empty_provider(self) -> InstanceProvider[InstanceType, KT, DT, VT, RT]:
        """Use this method to create an empty `InstanceProvider`

        Returns
        -------
        InstanceProvider[InstanceType, KT, DT, VT, RT]
            The newly created provider
        """        
        raise NotImplementedError

    @property
    @abstractmethod
    def dataset(self) -> InstanceProvider[InstanceType, KT, DT, VT, RT]:
        """This property contains the `InstanceProvider` that contains
        the original dataset. This provider should include all original
        instances.

        Returns
        -------
        InstanceProvider[InstanceType, KT, DT, VT, RT]
            The dataset :class:`InstanceProvider`
        """        
        raise NotImplementedError

    
    
    @property
    @abstractmethod
    def all_instances(self) -> InstanceProvider[InstanceType, KT, DT, VT, RT]:
        """This provider should include all instances in all providers.
        If there are any synthethic datapoints constructed, 
        they should be also in here.

        Returns
        -------
        InstanceProvider[InstanceType, KT, DT, VT, RT]
            The all_instances :class:`InstanceProvider`
        """        
        raise NotImplementedError

    @property
    def all_datapoints(self) -> InstanceProvider[InstanceType, KT, DT, VT, RT]:
        """This provider should include all instances in all providers.
        If there are any synthethic datapoints constructed, 
        they should be also in here.

        Returns
        -------
        InstanceProvider[InstanceType, KT, DT, VT, RT]
            The all_datapoints :class:`InstanceProvider`

        Warning
        -------
        Deprecated, use the all_instances property instead!

        """
        warnings.warn("Use the `all_instances` property instead!", category=DeprecationWarning)      
        return self.all_instances

    @property
    @abstractmethod
    def labels(self) -> LabelProvider[KT, LT]:
        """This property contains provider that has a mapping from instances to labels and
        vice-versa. 

        Returns
        -------
        LabelProvider[KT, LT]
            The label provider
        """        
        raise NotImplementedError

    def add_vectors(self, keys: Sequence[KT], vectors: Sequence[VT]) -> None:
        """This method adds feature vectors or embeddings to instances 
        associated with the keys in the first parameters. The sequences
        `keys` and `vectors` should have the same length.

        Parameters
        ----------
        keys : Sequence[KT]
            A sequence of keys
        vectors : Sequence[VT]
            A sequence of vectors that should be associated with the instances 
            of the sequence `keys`
        
        """        
        self.all_instances.bulk_add_vectors(keys, vectors)

    def create(self, *args: Any, **kwargs: Any) -> InstanceType:
        """Create a new Instance

        Returns
        -------
        InstanceType
            A new instance
        """        
        new_instance = self.all_instances.create(*args, **kwargs)
        return new_instance

    @abstractmethod
    def create_bucket(self, keys: Iterable[KT]) -> InstanceProvider[InstanceType, KT, DT, VT, RT]:
        """Create an InstanceProvider that contains certain keys found in this 
        environment.

        Parameters
        ----------
        keys : Iterable[KT]
            The keys that should be included in this bucket

        Returns
        -------
        InstanceProvider[InstanceType, KT, DT, VT, RT]
            An InstanceProvider that contains the instances specified in `keys`

        """        
        raise NotImplementedError

    def train_test_split(self, 
                         source: InstanceProvider[InstanceType, KT, DT, VT, RT],
                         train_size: Union[float, int]
                         ) -> Tuple[InstanceProvider[InstanceType, KT, DT, VT, RT], 
                                    InstanceProvider[InstanceType, KT, DT, VT, RT]]:
        """Divide an InstanceProvider into two different providers containing a random 
        division of the input according to the parameter `train_size`.

        Parameters
        ----------
        source : InstanceProvider[InstanceType, KT, DT, VT, RT]
            The InstanceProvider that should be divided
        train_size : Union[float, int]
            The number (int) of instances that should be included in the training
            or a float (between 0 and 1) of train / test ratio.

        Examples
        --------
        Example usage

        >>> train_val, test = env.train_test_split(provider, 0.70)
        >>> train, val = env.train_test_split(train_val, 0.70)


        Returns
        -------
        Tuple[InstanceProvider[InstanceType, KT, DT, VT, RT], InstanceProvider[InstanceType, KT, DT, VT, RT]]
            A Tuple containing two InstanceProviders:
                - The training set (containing `train_size` documents)
                - The test set
        """        
        if isinstance(train_size, float):
            n_train_docs = round(train_size*len(source))
        else:
            n_train_docs = train_size
        source_keys = list(frozenset(source.key_list))
        
        # Randomly sample train keys
        train_keys = random.sample(source_keys, n_train_docs)
        # The remainder should be used for testing        
        test_keys = frozenset(source_keys).difference(train_keys)
        
        train_provider = self.create_bucket(train_keys)
        test_provider = self.create_bucket(test_keys)
        return train_provider, test_provider

    def combine(self, 
                *providers: InstanceProvider[InstanceType, KT, DT, VT, RT],
                ) -> InstanceProvider[InstanceType, KT, DT, VT, RT]:
        """Combine Providers into a single Provider

        Parameters
        ----------
        providers
            The providers that should be combined into a single provider

        Returns
        -------
        InstanceProvider[InstanceType, KT, DT, VT, RT]
            The provider that contains all elements of the supplied Providers
        """        
                
        keys = union(*(frozenset(pr.key_list) for pr in providers))
        combined_provider = self.create_bucket(keys)
        return combined_provider

    def get_children(self, parent: Union[KT, Instance[KT, DT, VT, RT]]) -> InstanceProvider[InstanceType, KT, DT, VT, RT]:
        """Get the children that are registered to this parent

        Parameters
        ----------
        parent : Union[KT, Instance[KT, DT, VT, RT]]
            The parent from which you want to get the children from.

        Returns
        -------
        InstanceProvider[InstanceType, KT, DT, VT, RT]
            A Provider that contains all children
        """        
        child_keys = self.all_instances.get_children_keys(parent)
        new_bucket = self.create_bucket(child_keys)
        return new_bucket

    def get_parent(self, child: Union[KT, Instance[KT, DT, VT, RT]]) -> InstanceType:
        """Get the parent of a child

        Parameters
        ----------
        child : Union[KT, Instance[KT, DT, VT, RT]]
            A child instance from which you want to get the children from.

        Returns
        -------
        InstanceType
            The parent of this child instance
        """        
        return self.all_instances.get_parent(child)

    def discard_children(self, parent: Union[KT, Instance[KT, DT, VT, RT]]) -> None:
        """Discard all children from this parent

        Parameters
        ----------
        parent : Union[KT, Instance[KT, DT, VT, RT]]
            The parent Instance
        """        
        self.all_instances.discard_children(parent)

                                

class AbstractEnvironment(Environment[InstanceType, KT, DT, VT, RT, LT], 
                  ABC, Generic[InstanceType, KT, DT, VT, RT, LT]):
    """Environments provide an interface that enable you to access all data stored in the datasets.
    If there are labels stored in the environment, you can access these as well from here.

    There are two important properties in every :class:`Environment`:

    - :meth:`dataset`: Contains all Instances of the original dataset
    - :meth:`labels`: Contains an object that allows you to access labels easily

    Besides these properties, this object also provides methods to create new 
    :class:`~instancelib.InstanceProvider` objects that contain a subset of 
    the set of all instances stored in this environment.

    Examples
    --------

    Access the dataset:

    >>> dataset = env.dataset
    >>> instance = next(iter(dataset.values()))

    Access the labels:

    >>> labels = env.labels
    >>> ins_lbls = labels.get_labels(instance)

    Create a train-test split on the dataset (70 % train, 30 % test):

    >>> train, test = env.train_test_split(dataset, 0.70)
    """    
    pass