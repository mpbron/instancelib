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
from abc import ABC

from typing import Generic, Iterable, Dict, TypeVar, Any

from ..instances.base import Instance, InstanceProvider
from ..instances.memory import MemoryBucketProvider
from ..labels.memory import MemoryLabelProvider

from .base import AbstractEnvironment

from ..typehints import KT, DT, VT, RT, LT



InstanceType = TypeVar("InstanceType", bound="Instance[Any, Any, Any, Any]", covariant=True)

class AbstractMemoryEnvironment(
        AbstractEnvironment[InstanceType, KT, DT, VT, RT, LT],
        ABC, Generic[InstanceType, KT, DT, VT, RT, LT]):
    """Environments provide an interface that enable you to access all data stored in the datasets.
    If there are labels stored in the environment, you can access these as well from here.

    There are two important properties in every :class:`Environment`:

    - :meth:`dataset`: Contains all Instances of the original dataset
    - :meth:`labels`: Contains an object that allows you to access labels easily

    Besides these properties, this object also provides methods to create new 
    :class:`~instancelib.InstanceProvider` objects that contain a subset of 
    the set of all instances stored in this environment.

    Attributes
    ----------
    _public_dataset 
        An :class:`InstanceProvider` that contains all original Instances
    _dataset
        An :class:`InstanceProvider` that contains all instances
    _labelprovider
        This object contains all labels
    _named_provider
        All user generated providers that were given a name


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

    _public_dataset: InstanceProvider[InstanceType, KT, DT, VT, RT]
    """An :class:`~instancelib.InstanceProvider` that contains all original Instances"""
    _dataset: InstanceProvider[InstanceType, KT, DT, VT, RT]
    """An :class:`InstanceProvider` that contains all instances"""
    _labelprovider: MemoryLabelProvider[KT, LT]
    """This object contains all labels"""
    _named_providers: Dict[str, InstanceProvider[InstanceType, KT, DT, VT, RT]] = dict()
    """All user generated providers that were given a name"""

    @property
    def dataset(self) -> InstanceProvider[InstanceType, KT, DT, VT, RT]:
        return self._public_dataset

    @property
    def all_instances(self) -> InstanceProvider[InstanceType, KT, DT, VT, RT]:
        return self._dataset
    
    @property
    def labels(self) -> MemoryLabelProvider[KT, LT]:
        return self._labelprovider

    def create_bucket(self, keys: Iterable[KT]) -> InstanceProvider[InstanceType, KT, DT, VT, RT]:
        return MemoryBucketProvider[InstanceType, KT, DT, VT, RT](self._dataset, keys)

    def create_empty_provider(self) -> InstanceProvider[InstanceType, KT, DT, VT, RT]:
        return self.create_bucket([])

    def set_named_provider(self, name: str, value: InstanceProvider[InstanceType, KT, DT, VT, RT]):
        self._named_providers[name] = value

    def create_named_provider(self, name: str) -> InstanceProvider[InstanceType, KT, DT, VT, RT]:
        self._named_providers[name] = self.create_empty_provider()
        return self._named_providers[name]   
    
class MemoryEnvironment(
    AbstractMemoryEnvironment[InstanceType, KT, DT, VT, RT, LT],
        Generic[InstanceType, KT, DT, VT, RT, LT]):
    """This class implements the :class:`~abc.ABC` :class:`~instancelib.Environment`.
    In this method, all data is loaded and stored in RAM and is not preserved on disk.
    There are two important properties in every :class:`Environment`:

    - :meth:`dataset`: Contains all Instances of the original dataset
    - :meth:`labels`: Contains an object that allows you to access labels easily

    Besides these properties, this object also provides methods to create new 
    :class:`~instancelib.InstanceProvider` objects that contain a subset of 
    the set of all instances stored in this environment.

    Parameters
    ----------
    dataset : InstanceProvider[InstanceType, KT, DT, VT, RT]
        An InstanceProvider that contains all Instances
    labelprovider : MemoryLabelProvider[KT, LT]
        The label provider that contains the labels associated
        with the instances from the :data:`dataset` variable

    Attributes
    ----------
    _public_dataset 
        An :class:`InstanceProvider` that contains all original Instances
    _dataset
        An :class:`InstanceProvider` that contains all instances
    _labelprovider
        This object contains all labels
    _named_provider
        All user generated providers that were given a name

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

    Store the environment to disk:

    >>> import pickle
    >>> with open("file.pkl", "wb") as fh:
    ...     pickle.dump(env, fh)
    >>> print("The file is saved to file.pkl")

    Load the environment from disk:

    >>> import pickle
    >>> with open("file.pkl", "rb") as fh:
    ...     env = pickle.load(fh)
    >>> dataset = env.dataset
    """    
    
    def __init__(
            self,
            dataset: InstanceProvider[InstanceType, KT, DT, VT, RT],
            labelprovider: MemoryLabelProvider[KT, LT]
        ):
        """[summary]

        Parameters
        ----------
        dataset : InstanceProvider[InstanceType, KT, DT, VT, RT]
            [description]
        labelprovider : MemoryLabelProvider[KT, LT]
            [description]
        """        
        self._dataset = dataset
        self._public_dataset = MemoryBucketProvider[InstanceType, KT, DT, VT, RT](dataset, dataset.key_list)
        self._labelprovider = labelprovider
        self._named_providers = dict()
    
    

    



    

    
    
    



        

