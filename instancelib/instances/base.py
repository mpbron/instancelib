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

from typing import (Any, Callable, Generic, Iterable, Iterator, List, MutableMapping,
                    Optional, Sequence, Tuple, TypeVar, Union)

import numpy as np  # type: ignore

from ..utils.chunks import divide_iterable_in_lists
from ..utils.func import filter_snd_none_zipped

from ..typehints import KT, DT, VT, RT

_V = TypeVar("_V")

class Instance(ABC, Generic[KT, DT, VT, RT]):

    @property
    @abstractmethod
    def data(self) -> DT:
        """Return the raw data of this instance


        Returns
        -------
        DT
            The Raw Data
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def representation(self) -> RT:
        """Return a representation for annotation


        Returns
        -------
        RT
            A representation of the raw data
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def vector(self) -> Optional[VT]:
        """Get the vector represenation of the raw data

        Returns
        -------
        Optional[VT]
            The Vector
        """
        raise NotImplementedError

    @vector.setter
    def vector(self, value: Optional[VT]) -> None:  # type: ignore
        raise NotImplementedError

    @property
    @abstractmethod
    def identifier(self) -> KT:
        """Get the identifier of the instance

        Returns
        -------
        KT
            The identifier key of the instance
        """
        raise NotImplementedError

    @identifier.setter
    def identifier(self, value: KT) -> None:
        raise NotImplementedError

    def __str__(self) -> str:
        str_rep = f"Instance:\n Identifier => {self.identifier} \n Data => {self.data} \n Vector present => {self.vector is not None}"
        return str_rep

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def map_data(func: Callable[[DT], _V]) -> Callable[[Instance[KT, DT, VT, RT]], _V]:
        def wrapped(instance: Instance[KT, DT, VT, RT]) -> _V:
            return(func(instance.data))
        return wrapped

    @staticmethod
    def map_vector(func: Callable[[VT], _V]) -> Callable[[Instance[KT, DT, VT, RT]], Optional[_V]]:
        def wrapped(instance: Instance[KT, DT, VT, RT]) -> Optional[_V]:
            if instance.vector is not None:
                return(func(instance.vector))
            return None
        return wrapped

    @staticmethod
    def vectorized_data_map(
        func: Callable[[Iterable[DT]], _V]
        ) -> Callable[[Iterable[Instance[KT, DT, VT, RT]]], _V]:
        def wrapped(instances: Iterable[Instance[KT, DT, VT, RT]]) -> _V:
            data = (instance.data for instance in instances)
            results = func(data)
            return results
        return wrapped

InstanceType = TypeVar("InstanceType", bound="Instance[Any, Any, Any, Any]")

class InstanceProvider(MutableMapping[KT, InstanceType], 
                       ABC, Generic[InstanceType, KT, DT, VT, RT]):
    """[summary]

    Parameters
    ----------
    MutableMapping : [type]
        [description]
    ABC : [type]
        [description]
    Generic : [type]
        [description]

    Returns
    -------
    [type]
        [description]

    Yields
    -------
    [type]
        [description]
    """

    def add_child(self,
                  parent: Union[KT, InstanceType],
                  child: Union[KT, InstanceType]) -> None:
        raise NotImplementedError

    def get_children(self, parent: Union[KT, InstanceType]) -> Sequence[InstanceType]:
        raise NotImplementedError

    def get_parent(self, child: Union[KT, InstanceType]) -> InstanceType:
        raise NotImplementedError

    @abstractmethod
    def __contains__(self, item: object) -> bool:
        """Special method that checks if something is contained in this 
        provider.

        Parameters
        ----------
        item : object
            The item of which we want to know if it is contained in this
            provider

        Returns
        -------
        bool
            True if the provider contains `item`. 

        Examples
        --------
        Example usage; check if the item exists and then remove it

        >>> doc_id = 20
        >>> provider = InstanceProvider()
        >>> if doc_id in provider:
        ...     del provider[doc_id]
        """
        raise NotImplementedError

    @abstractmethod
    def __iter__(self) -> Iterator[KT]:
        """Enables you to iterate over Instances

        Yields
        -------
        Iterator[KT]
            [description]

        Raises
        ------
        NotImplementedError
            [description]
        """
        raise NotImplementedError

    def add(self, instance: Instance[KT, DT, VT, RT]) -> None:
        """Add an instance to this provider. If the 
        provider already contains `instance`, nothing happens.

        Parameters
        ----------
        instance : Instance[KT, DT, VT, RT]
            The instance that should be added to the provider
        """
        self.__setitem__(instance.identifier, instance)

    def add_range(self, *instances: Instance[KT, DT, VT, RT]) -> None:
        """Add multiple instances to this provider. If the 
        provider already contains `instance`, nothing happens.

        Parameters
        ----------
        instance : Instance[KT, DT, VT, RT]
            The instance that should be added to the provider
        """
        for instance in instances:
            self.add(instance)

    def discard(self, instance: Instance[KT, DT, VT, RT]) -> None:
        """Remove an instance from this provider. If the 
        provider does not contain `instance`, nothing happens.

        Parameters
        ----------
        instance : Instance[KT, DT, VT, RT]
            The instance that should be removed from the provider
        """
        try:
            self.__delitem__(instance.identifier)
        except KeyError:
            pass  # To adhere to Set.discard(...) behavior

    @property
    def key_list(self) -> List[KT]:
        """Return a list of all instance keys in this provider

        Returns
        -------
        List[KT]
            A list of instance keys
        """
        return list(self.keys())

    @property
    @abstractmethod
    def empty(self) -> bool:
        """Determines if the provider does not contain instances

        Returns
        -------
        bool
            True if the provider is empty
        """
        raise NotImplementedError

    @abstractmethod
    def get_all(self) -> Iterator[InstanceType]:
        """Get an iterator that iterates over all instances

        Yields
        ------
        Instance[KT, DT, VT, RT]
            An iterator that iterates over all instances
        """
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """Removes all instances from the provider

        Warning
        -------
        Use this operation with caution! This operation is intended for
        use with providers that function as temporary user queues, not
        for large proportions of the dataset like `unlabeled` and `labeled`
        sets.
        """
        raise NotImplementedError

    def bulk_add_vectors(self, keys: Sequence[KT], values: Sequence[VT]) -> None:
        """This methods adds vectors in `values` to the instances specified
        in `keys`. 

        In some use cases, vectors are not known beforehand. This library
        provides several :term:`vectorizer` s that convert raw data points
        in feature vector form. Once these vectors are available, they can be 
        added to the provider by using this method

        Parameters
        ----------
        keys : Sequence[KT]
            A sequence of keys
        values : Sequence[VT]
            A sequence of vectors

        Warning
        -------
        We assume that the indices and length of the parameters `keys` and `values`
        match.
        """
        for key, vec in zip(keys, values):
            self[key].vector = vec

    def bulk_get_vectors(self, keys: Sequence[KT]) -> Tuple[Sequence[KT], Sequence[VT]]:
        """Given a list of instance `keys`, return the vectors

        Parameters
        ----------
        keys : Sequence[KT]
            A list of vectors

        Returns
        -------
        Tuple[Sequence[KT], Sequence[VT]]
            A tuple of two sequences, one with `keys` and one with `vectors`.
            The indices match, so the instance with ``keys[2]`` has as
            vector ``vectors[2]``

        Warning
        -------
        Some underlying implementations do not preserve the ordering of the parameter
        `keys`. Therefore, always use the keys variable from the returned tuple for 
        the correct matching.
        """
        vector_pairs = ((key, self[key].vector) for key in keys)
        ret_keys, ret_vectors = filter_snd_none_zipped(vector_pairs)
        return ret_keys, ret_vectors  # type: ignore

    def data_chunker(self, batch_size: int) -> Iterator[Sequence[InstanceType]]:
        """Iterate over all instances (with or without vectors) in 
        this provider

        Parameters
        ----------
        batch_size : int
            The batch size, the generator will return lists with size `batch_size`

        Yields
        -------
        Sequence[Instance[KT, DT, VT, RT]]]
            A sequence of instances with length `batch_size`. The last list may have
            a shorter length.
        """
        chunks = divide_iterable_in_lists(self.values(), batch_size)
        yield from chunks

    def vector_chunker_selector(self, keys: Iterable[KT], batch_size: int) -> Iterator[Sequence[Tuple[KT, VT]]]:
        included_ids = frozenset(self.key_list).intersection(keys)
        id_vecs = ((elem.identifier, elem.vector)
                   for elem in self.values() if elem.vector is not None and elem.identifier in included_ids)
        chunks = divide_iterable_in_lists(id_vecs, batch_size)
        return chunks

    def vector_chunker(self, batch_size: int) -> Iterator[Sequence[Tuple[KT, VT]]]:
        """Iterate over all pairs of keys and vectors in 
        this provider

        Parameters
        ----------
        batch_size : int
            The batch size, the generator will return lists with size `batch_size`

        Returns
        -------
        Iterator[Sequence[Tuple[KT, VT]]]
            An iterator over sequences of key vector tuples

        Yields
        -------
        Iterator[Sequence[Tuple[KT, VT]]]
            Sequences of key vector tuples
        """
        id_vecs = ((elem.identifier, elem.vector)
                   for elem in self.values() if elem.vector is not None)
        chunks = divide_iterable_in_lists(id_vecs, batch_size)
        return chunks

    def bulk_get_all(self) -> List[InstanceType]:
        """Returns a list of all instances in this provider.

        Returns
        -------
        List[Instance[KT, DT, VT, RT]]
            A list of all instances in this provider

        Warning
        -------
        When using this method on very large providers with lazily loaded instances, this
        may yield Out of Memory errors, as all the data will be loaded into RAM.
        Use with caution!
        """
        return list(self.get_all())

    def map_mutate(self, func: Callable[[InstanceType], InstanceType]) -> None:
        keys = self.key_list
        for key in keys:
            instance = self[key]
            upd_instance = func(instance)
            self[key] = upd_instance

    def map(self, func: Callable[[InstanceType], _V]) -> Iterator[_V]:
        keys = self.key_list
        for key in keys:
            instance = self[key]
            result = func(instance)
            yield result

    def data_map(self, func: Callable[[DT], _V]) -> Iterator[_V]:
        instances = self.values()
        mapped_f = Instance[KT, DT, VT, RT].map_data(func)
        results = map(mapped_f, instances)
        yield from results

    def all_data(self) -> Iterator[DT]:
        yield from (instance.data for instance in self.values())

    def vectorized_map(self, 
                       func: Callable[[Iterable[InstanceType]], _V], 
                       batch_size: int = 200) -> Iterator[_V]:
        chunks = divide_iterable_in_lists(self.values(), batch_size)
        results = map(func, chunks)
        yield from results

    def vectorized_data_map(self, 
                            func: Callable[[Iterable[DT]], _V],
                            batch_size: int = 200
                            ) -> Iterator[_V]:
        chunks = divide_iterable_in_lists(self.values(), batch_size)
        mapped_f = Instance[KT, DT, VT, RT].vectorized_data_map(func)
        results = map(mapped_f, chunks)
        yield from results

    @abstractmethod
    def create(self, *args: Any, **kwargs: Any) -> InstanceType:
        raise NotImplementedError

class AbstractBucketProvider(InstanceProvider[InstanceType, KT, DT, VT, RT], ABC, 
                                   Generic[InstanceType, KT, DT, VT, RT]):
    dataset: InstanceProvider[InstanceType, KT, DT, VT, RT]
    
    @abstractmethod
    def _add_to_bucket(self, key: KT) -> None:
        raise NotImplementedError

    @abstractmethod
    def _remove_from_bucket(self, key: KT) -> None:
        raise NotImplementedError

    @abstractmethod
    def _in_bucket(self, key: KT) -> bool:
        raise NotImplementedError
    
    @abstractmethod
    def _clear_bucket(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _len_bucket(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def _bucket(self) -> Iterable[KT]:
        raise NotImplementedError

    def __iter__(self) -> Iterator[KT]:
        yield from self._bucket

    def __getitem__(self, key: KT):
        if self._in_bucket(key):
            return self.dataset[key]
        raise KeyError(
            f"This datapoint with key {key} does not exist in this provider")

    def __setitem__(self, key: KT, value: InstanceType) -> None:
        self._add_to_bucket(key)
        self.dataset[key] = value  # type: ignore

    def __delitem__(self, key: KT) -> None:
        self._remove_from_bucket(key)

    def __len__(self) -> int:
        return self._len_bucket()

    def __contains__(self, key: object) -> bool:
        return self._in_bucket(key)

    def get_all(self) -> Iterator[InstanceType]:
        yield from list(self.values())

    def vector_chunker(self, batch_size: int) -> Iterator[Sequence[Tuple[KT, VT]]]:
        results = self.dataset.vector_chunker_selector(self.key_list, batch_size)
        return results

    def clear(self) -> None:
        self._clear_bucket()

    @property
    def empty(self) -> bool:
        return not self._bucket

    def add_child(self, 
                  parent: Union[KT, InstanceType], 
                  child: Union[KT, InstanceType]) -> None:
        self.dataset.add_child(parent, child)

    def get_children(self, parent: Union[KT, InstanceType]) -> Sequence[InstanceType]:
        return self.dataset.get_children(parent)

    def get_parent(self, child: Union[KT, InstanceType]) -> InstanceType:
        return self.dataset.get_parent(child)

    def create(self, *args: Any, **kwargs: Any) -> InstanceType:
        new_instance = self.dataset.create(*args, **kwargs)
        self.add(new_instance)
        return new_instance