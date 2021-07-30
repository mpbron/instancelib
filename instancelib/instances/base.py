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

from ..utils.chunks import divide_iterable_in_lists
from ..utils.func import filter_snd_none_zipped

from ..typehints import KT, DT, VT, RT

_V = TypeVar("_V")

class Instance(ABC, Generic[KT, DT, VT, RT]):
    """A base Instance Class. 

    Every Instance contains 4 properties:

        - A unique identifier (`identifier`)
        - The raw data (`data`)
        - A vector representation of the data (`vector`)
        - A human readable representation (`representation`)

    The ABC Instance has four Generic types:

        - :data:`~instancelib.typehints.KT`: The type of the key
        - :data:`~instancelib.typehints.DT`: The type of the data
        - :data:`~instancelib.typehints.VT`: The type of the vector
        - :data:`~instancelib.typehints.RT`: The type of the representation

    Combining these four items in a single object enables easy transfer between
    different operations like predictions, annotatation and transformation.
    """    

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
        """Set the vector representation of the raw data

        Parameters
        ----------
        value : Optional[VT]
            A vector value (this may be `None`)

        Note
        ----
        It may be better to use the 
        :meth:`InstanceProvider.bulk_add_vectors` method
        if you want update the vectors of many instances
        """        
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
        """Set the identifier of the instance

        Parameters
        ----------
        value : KT
            The new identifier
        """        
        raise NotImplementedError

    def __str__(self) -> str:
        str_rep = f"Instance:\n Identifier => {self.identifier} \n Data => {self.data} \n Vector present => {self.vector is not None}"
        return str_rep

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def map_data(func: Callable[[DT], _V]) -> Callable[[Instance[KT, DT, VT, RT]], _V]:
        """Transform function that works on raw data into a function that works on
        :class:`Instance` objects.

        Parameters
        ----------
        func : Callable[[DT], _V]
            The function that works on raw data

        Returns
        -------
        Callable[[Instance[KT, DT, VT, RT]], _V]
            The transformed function
        """        
        def wrapped(instance: Instance[KT, DT, VT, RT]) -> _V:
            return(func(instance.data))
        return wrapped

    @staticmethod
    def map_vector(func: Callable[[VT], _V]) -> Callable[[Instance[KT, DT, VT, RT]], Optional[_V]]:
        """Transform function that works on vectors into a function that works on
        :class:`Instance` objects.

        Parameters
        ----------
        func : Callable[[VT], _V]
            The function that works on vectors

        Returns
        -------
        Callable[[Instance[KT, DT, VT, RT]], _V]
            The transformed function
        """        
        def wrapped(instance: Instance[KT, DT, VT, RT]) -> Optional[_V]:
            if instance.vector is not None:
                return(func(instance.vector))
            return None
        return wrapped

    @staticmethod
    def vectorized_data_map(
        func: Callable[[Iterable[DT]], _V]
        ) -> Callable[[Iterable[Instance[KT, DT, VT, RT]]], _V]:
        """Transform function that works on sequences of raw data  
        into a function that works on sequences of :class:`Instance` objects.

        Parameters
        ----------
        func : Callable[[Iterable[DT]], _V]
            The function that works on sequences of raw data

        Returns
        -------
        Callable[[Iterable[Instance[KT, DT, VT, RT]]], _V]
            The transformed function
        """        
        def wrapped(instances: Iterable[Instance[KT, DT, VT, RT]]) -> _V:
            data = (instance.data for instance in instances)
            results = func(data)
            return results
        return wrapped

InstanceType = TypeVar("InstanceType", bound="Instance[Any, Any, Any, Any]")

class InstanceProvider(MutableMapping[KT, InstanceType], 
                       ABC, Generic[InstanceType, KT, DT, VT, RT]):
    """The Base InstanceProvider class.

    This class provides an abstract implementation for a dataset.
    The InstanceProvider has five Generic types:

        - :data:`InstanceType` : A subclass of :class:`Instance`
        - :data:`~instancelib.typehints.KT`: The type of the key
        - :data:`~instancelib.typehints.DT`: The type of the data
        - :data:`~instancelib.typehints.VT`: The type of the vector
        - :data:`~instancelib.typehints.RT`: The type of the representation

    Specifying these allows Python to ensure the correctness of your implementation 
    and eases further integration in your application.

    Examples
    --------
    
    Instance access:

    >>> provider = InstanceProvider() # Replace with your implementation's constructor
    >>> first_key = next(iter(textprovider))
    >>> first_doc = textprovider[first_key]

    Set operations:

    >>> new_instance = Instance()
    >>> provider.add(new_instance)
    >>> provider.discard(new_instance)

    Example implementation:
    
    >>> class TextProvider(InstanceProvider[Instance[int, str, np.ndarray, str],
    ...                                     int, str, np.ndarray, str]):
    ...     # Further implementation is needed
    >>> textprovider = TextProvider()

    There are a number of :func:`~abc.abstractmethod` that need to be implemented
    in your own implementation. See the source of this file to see what you need to
    implement.
    """

    @abstractmethod
    def add_child(self,
                  parent: Union[KT, Instance[KT, DT, VT, RT]],
                  child: Union[KT, Instance[KT, DT, VT, RT]]) -> None:
        """Register a parent child relation between two instances

        Parameters
        ----------
        parent : Union[KT, Instance[KT, DT, VT, RT]]
            The parent instance (or identifier)
        child : Union[KT, Instance[KT, DT, VT, RT]]
            The child instance (or identifier)
        """        
        raise NotImplementedError

         
    @abstractmethod
    def get_children(self, parent: Union[KT, Instance[KT, DT, VT, RT]]) -> Sequence[InstanceType]:
        """Get the children that are registered to this parent

        Parameters
        ----------
        parent : Union[KT, Instance[KT, DT, VT, RT]]
            The parent from which you want to get the children from.

        Returns
        -------
        Sequence[InstanceType]
            A list containing the children
        """        
        raise NotImplementedError

    @abstractmethod
    def discard_children(self, parent: Union[KT, Instance[KT, DT, VT, RT]]) -> None:
        """Discard the children that are registered to this parent

        Parameters
        ----------
        parent : Union[KT, Instance[KT, DT, VT, RT]]
            The parent from which you want to get the children from.
        """
        raise NotImplementedError        

    def get_children_keys(self, parent: Union[KT, Instance[KT, DT, VT, RT]]) -> Sequence[KT]:
        """Get the children that are registered to this parent

        Parameters
        ----------
        parent : Union[KT, Instance[KT, DT, VT, RT]]
            The parent from which you want to get the children from.

        Returns
        -------
        Sequence[InstanceType]
            A list containing the children
        """
        child_keys = [ins.identifier for ins in self.get_children(parent)]
        return child_keys

    @abstractmethod
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

        Raises
        ------
        KeyError
            If there is no parent associated with this :class:`Instance`
        """        
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
        ------
        :class:`KT`
            Keys included in the provider 
        """
        raise NotImplementedError

    def add(self, instance: Instance[KT, DT, VT, RT]) -> None:
        """Add an instance to this provider. 
        
        If the provider already contains `instance`, nothing happens.

        Parameters
        ----------
        instance : Instance[KT, DT, VT, RT]
            The instance that should be added to the provider
        """
        self.__setitem__(instance.identifier, instance)

    def add_range(self, *instances: Instance[KT, DT, VT, RT]) -> None:
        """Add multiple instances to this provider. 
        
        If the provider already contains `instance`, nothing happens.

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
        InstanceType
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
        keys
            A sequence of keys
        values
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
    
    def data_chunker(self, batch_size: int = 200) -> Iterator[Sequence[Tuple[KT, DT]]]:
        """Iterate over all instances data parts in 
        this provider

        Parameters
        ----------
        batch_size : int
            The batch size, the generator will return lists with size `batch_size`

        Yields
        -------
        Sequence[Tuple[KT,DT]]
            A sequence of instances with length `batch_size`. The last list may have
            a shorter length.
        """
        datapoints = ((ins.identifier, ins.data) for ins in self.values())
        chunks = divide_iterable_in_lists(datapoints, batch_size)
        yield from chunks

    def data_chunker_selector(self, keys: Iterable[KT], batch_size: int = 200) -> Iterator[Sequence[Tuple[KT, DT]]]:
        keyset = frozenset(keys)
        datapoints = ((ins.identifier, ins.data) for ins in self.values() if ins.identifier in keyset)
        chunks = divide_iterable_in_lists(datapoints, batch_size)
        yield from chunks
    

    def instance_chunker(self, batch_size: int = 200) -> Iterator[Sequence[InstanceType]]:
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

    def vector_chunker_selector(self, keys: Iterable[KT], batch_size: int = 200) -> Iterator[Sequence[Tuple[KT, VT]]]:
        """Iterate over all instances (with or without vectors) in belonging the identifier 
        :class:`Iterable` in the `keys` parameter.

        Parameters
        ----------
        keys : Iterable[KT]
            The keys that should should be chunked

        batch_size : int
            The batch size, the generator will return lists with size `batch_size`

        Yields
        -------
        Sequence[Instance[KT, DT, VT, RT]]]
            A sequence of instances with length `batch_size`. The last list may have
            a shorter length.

        Returns
        -------
        Iterator[Sequence[Tuple[KT, VT]]]
            An iterator over sequences of key vector tuples
        """
        included_ids = frozenset(self.key_list).intersection(keys)
        id_vecs = ((elem.identifier, elem.vector)
                   for elem in self.values() if elem.vector is not None and elem.identifier in included_ids)
        chunks = divide_iterable_in_lists(id_vecs, batch_size)
        return chunks

    def vector_chunker(self, batch_size: int = 200) -> Iterator[Sequence[Tuple[KT, VT]]]:
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
        Sequence[Tuple[KT, VT]]
            Sequences of key vector tuples
        """
        yield from self.vector_chunker_selector(self.key_list, batch_size)

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

    def map_mutate(self, func: Callable[[InstanceType], Instance[KT, DT, VT, RT]]) -> None:
        """Run a function on this provider that modifies all Instances in place

        Parameters
        ----------
        func : Callable[[InstanceType], InstanceType]
            A function that modifies instances in place
        """        
        keys = self.key_list
        for key in keys:
            instance = self[key]
            upd_instance = func(instance)
            self[key] = upd_instance

    def map(self, func: Callable[[InstanceType], _V]) -> Iterator[_V]:
        """A higher order function that maps any function that works on
        individual :class:`Instance` objects on every contained object in
        this provider. 

        Parameters
        ----------
        func : Callable[[InstanceType], _V]
            A function that works on :class:`Instance` objects of type `InstanceType`

        Yields
        -------
        Iterator[_V]
            The values produced by the function `func`
        """        
        keys = self.key_list
        for key in keys:
            instance = self[key]
            result = func(instance)
            yield result

    def data_map(self, func: Callable[[DT], _V]) -> Iterator[_V]:
        """A higher order function that maps any function that works on
        individual :class:`~instancelib.typehints.KT` object 
        on every  :class:`Instance` object in this provider. 

        Parameters
        ----------
        func
            The function that should be applied

        Yields
        -------
        _V
            The values produced by the function `func`
        """
        instances = self.values()
        mapped_f = Instance[KT, DT, VT, RT].map_data(func)
        results = map(mapped_f, instances)
        yield from results

    def all_data(self) -> Iterator[DT]:
        """Return all the raw data from the instances in this provider

        Yields
        ------
        DT
            Raw data
        """        
        yield from (instance.data for instance in self.values())

    def vectorized_map(self, 
                       func: Callable[[Iterable[InstanceType]], _V], 
                       batch_size: int = 200) -> Iterator[_V]:
        """Maps a function that works on multiple instances
        onto all the instances in batches of size `batch_size`.

        Note: If you run a function that combines multiple instances into
        a single result, this may possibly lead to undiserable results if 
        batches are not taken into account.

        Parameters
        ----------
        func : Callable[[Iterable[InstanceType]], _V]
            The function that should be applied
        batch_size : int, optional
            The size of the batch, by default 200

        Yields
        -------
        _V
            The result type of the function in parameter `func`
        """        
        chunks = divide_iterable_in_lists(self.values(), batch_size)
        results = map(func, chunks)
        yield from results

    def vectorized_data_map(self, 
                            func: Callable[[Iterable[DT]], _V],
                            batch_size: int = 200
                            ) -> Iterator[_V]:
        """Maps a function that works on multiple raw data points
        onto all the instances in batches of size `batch_size`.

        Note: If you run a function that combines multiple instances into
        a single result, this may possibly lead to undiserable results if 
        batches are not taken into account.

        Parameters
        ----------
        func
            The function that should be applied
        batch_size : int, optional
            The size of the batch, by default 200

        Yields
        -------
        _V
            The result type of the function in parameter `func`
        """        
        chunks = divide_iterable_in_lists(self.values(), batch_size)
        mapped_f = Instance[KT, DT, VT, RT].vectorized_data_map(func)
        results = map(mapped_f, chunks)
        yield from results

    @abstractmethod
    def create(self, *args: Any, **kwargs: Any) -> InstanceType:
        """Create a new instance of type :data:`InstanceType`.
        The created instance is subsequently added to the provider.
        
        Note: The number of arguments and keyword arguments may differ
        in actual implementation, so there are no standard arguments.

        Returns
        -------
        InstanceType
            The new instance Type
        """        
        raise NotImplementedError

class AbstractBucketProvider(InstanceProvider[InstanceType, KT, DT, VT, RT], ABC, 
                                   Generic[InstanceType, KT, DT, VT, RT]):
    """This class allows the creation of subsets (`buckets`) from a provider,
    without copying data, while still preserving the :class:`InstanceProvider`
    API.

    For example, in Poolbased Active Learning, the dataset is partitioned
    in several sets; e.g., the `labeled` and `unlabeled` parts of the dataset. 
    Or in traditional supervised learning, the train, test and validation sets.
    No data is copied, only a set of identifiers is kept in this provider.
    All data resides in the original provider.

    Attributes
    ----------
    dataset
        The :class:`InstanceProvider` that you want to take a subset from
    """    

    dataset: InstanceProvider[InstanceType, KT, DT, VT, RT]
    """The original dataset. All data will remain there"""    
    
    @abstractmethod
    def _add_to_bucket(self, key: KT) -> None:
        """Adds the :class:`Instance` with identifier `key` to the bucket

        Parameters
        ----------
        key : KT
            The identifier for the :class:`Instance` that should be added
        """        
        raise NotImplementedError

    @abstractmethod
    def _remove_from_bucket(self, key: KT) -> None:
        """Removes the :class:`Instance` with identifier `key` from the bucket

        Parameters
        ----------
        key : KT
            The identifier for the :class:`Instance` that should be removed
        """        
        raise NotImplementedError

    @abstractmethod
    def _in_bucket(self, key: KT) -> bool:
        """Returns if the :class:`Instance` with identifier `key` exists 
        within this bucket

        Parameters
        ----------
        key : KT
            The identifier for the :class:`Instance` that should be added
        """        
        raise NotImplementedError
    
    @abstractmethod
    def _clear_bucket(self) -> None:
        """Removes all elements from this bucket
        """        
        raise NotImplementedError

    @abstractmethod
    def _len_bucket(self) -> int:
        """Returns the number of elements in the buckets

        Returns
        -------
        int
            The size of the bucket
        """        
        raise NotImplementedError

    @property
    @abstractmethod
    def _bucket(self) -> Iterable[KT]:
        """Return an iterable of all identifiers in the bucket.

        Returns
        -------
        Iterable[KT]
            An :class:`Iterable` that contains all identifiers
            present in this bucket
        """        
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

    def vector_chunker(self, batch_size: int = 200) -> Iterator[Sequence[Tuple[KT, VT]]]:
        results = self.dataset.vector_chunker_selector(self.key_list, batch_size)
        return results

    def data_chunker(self, batch_size: int = 200) -> Iterator[Sequence[Tuple[KT, DT]]]:
        results = self.dataset.data_chunker_selector(self.key_list, batch_size)
        return results

    def data_chunker_selector(self, keys: Iterable[KT], batch_size: int = 200) -> Iterator[Sequence[Tuple[KT, DT]]]:
        keyset = frozenset(self.key_list).intersection(keys)
        results = self.dataset.data_chunker_selector(keyset, batch_size)
        return results

    def vector_chunker_selector(self, keys: Iterable[KT], batch_size: int = 200) -> Iterator[Sequence[Tuple[KT, VT]]]:
        keyset = frozenset(self.key_list).intersection(keys)
        results = self.dataset.vector_chunker_selector(keyset, batch_size)
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

    def get_children_keys(self, parent: Union[KT, Instance[KT, DT, VT, RT]]) -> Sequence[KT]:
        return self.dataset.get_children_keys(parent)

    def get_children(self, parent: Union[KT, InstanceType]) -> Sequence[InstanceType]:
        return self.dataset.get_children(parent)

    def get_parent(self, child: Union[KT, InstanceType]) -> InstanceType:
        return self.dataset.get_parent(child)

    def discard_children(self, parent: Union[KT, Instance[KT, DT, VT, RT]]) -> None:
        return self.dataset.discard_children(parent)

    def create(self, *args: Any, **kwargs: Any) -> InstanceType:
        new_instance = self.dataset.create(*args, **kwargs)
        self.add(new_instance)
        return new_instance

class SubtractionProvider(AbstractBucketProvider[InstanceType, KT, DT, VT, RT], 
                          ABC, 
                          Generic[InstanceType, KT, DT, VT, RT]):
    """This abstract class allows the creation of large subsets (`buckets`) that 
    do not contain some elements, specified in a `bucket`.
    No data is copied, however, the :class:`InstanceProvider` API is preserved.

    In some underlying implementations (like a Many to Many relation in Django),
    the creation of a large elements set takes a lot of time.
    This class allows the creation to subtract a (small) bucket from the dataset
    and include only the remainder.

    This method can be used in the Poolbased Active Learning setting; suppose
    you have a small `labeled` set and a huge dataset. 
    You can subtract the `labeled` from the dataset and create an InstanceProvider
    that contains all `unlabeled` examples. 

    Attributes
    ----------
    dataset
        The :class:`InstanceProvider` that you want to take a subset from

    bucket
        The :class:`InstanceProvider` that you want to exclude from the dataset

    Warning
    -------
    If possible, do not use this class: a solution that is based on only :class:`InstanceProvider` objects
    and :class:`AbstractBucketProvider` will probably be faster.
    """
    
    bucket: InstanceProvider[InstanceType, KT, DT, VT, RT]
    """The provider that should be excluded from the original `dataset`."""

    @property
    def _bucket(self) -> Iterable[KT]:
        ds_keys = frozenset(self.dataset)
        bu_keys = frozenset(self.bucket)
        difference = ds_keys.difference(bu_keys)
        return iter(difference)

    def _in_bucket(self, key: KT) -> bool:
        return key not in self.bucket and key in self.dataset

    def _add_to_bucket(self, key: KT) -> None:
        instance = self.dataset[key]
        self.bucket.discard(instance)

    def _remove_from_bucket(self, key: KT) -> None:
        instance = self.dataset[key]
        self.bucket.add(instance)

    def _clear_bucket(self) -> None:
        pass

    def _len_bucket(self) -> int:
        ds_keys = frozenset(self.dataset)
        bu_keys = frozenset(self.bucket)
        difference = ds_keys.difference(bu_keys)
        return len(difference)

    def create(self, *args: Any, **kwargs: Any) -> InstanceType:
        new_instance = self.dataset.create(*args, **kwargs)
        return new_instance

    def clear(self) -> None:
        pass


        