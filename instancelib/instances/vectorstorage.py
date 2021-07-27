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

import functools
from io import UnsupportedOperation
from abc import abstractmethod
from typing import (Any, Callable, Generic, Iterator, MutableMapping, Sequence,
                    Tuple, TypeVar, Union)


from ..typehints import KT, VT, MT

F = TypeVar("F", bound=Callable[..., Any])

class VectorStorage(MutableMapping[KT, VT], Generic[KT, VT, MT]):
    @property
    @abstractmethod
    def writeable(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, k: KT) -> VT:
        raise NotImplementedError

    @abstractmethod
    def __setitem__(self, k: KT, value: VT) -> None:
        raise NotImplementedError

    @abstractmethod
    def __contains__(self, item: object) -> bool:
        raise NotImplementedError

    @abstractmethod
    def __iter__(self) -> Iterator[KT]:
        raise NotImplementedError

    @abstractmethod
    def add_bulk(self, input_keys: Sequence[KT], input_values: Union[Sequence[VT]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_matrix(self, keys: Sequence[KT]) -> Tuple[Sequence[KT], MT]:
        raise NotImplementedError

    @abstractmethod
    def get_vectors(self, keys: Sequence[KT]) -> Tuple[Sequence[KT], Sequence[VT]]:
        """Return the vectors that correspond with the `keys` 

        Parameters
        ----------
        keys : Sequence[KT]
            A list of identifier keys

        Returns
        -------
        Tuple[Sequence[KT], Sequence[VT]]
            A tuple containing two lists:

                - A list with identifier (order may differ from `keys` argument)
                - A list with vectors 
        """        
        raise NotImplementedError


    @abstractmethod
    def matrices_chunker(self) -> Iterator[Tuple[Sequence[KT], MT]]:
        raise NotImplementedError

    @abstractmethod
    def get_vectors_chunked(self, keys: Sequence[KT], chunk_size: int = 200) -> Iterator[Tuple[Sequence[KT], Sequence[VT]]]:
        """Return vectors in chunks of `chunk_size` containing the vectors requested in `keys`

        Parameters
        ----------
        keys : Sequence[KT]
            A list of identifier keys
        chunk_size : int, optional
            The size of the chunks, by default 200

        Yields
        -------
        Tuple[Sequence[KT], Sequence[VT]]
            A tuple containing two lists:

                - A list with identifiers (order may differ from `keys` argument)
                - A list with vectors 
        """        
        raise NotImplementedError

    @abstractmethod
    def get_vectors_zipped(self, keys: Sequence[KT], chunk_size: int = 200) -> Iterator[Sequence[Tuple[KT, VT]]]:
        raise NotImplementedError

    @abstractmethod
    def __enter__(self) -> VectorStorage[KT, VT, MT]:
        raise NotImplementedError
    @abstractmethod
    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def reload(self) -> None:
        """Reload the index from disk
        """
        raise NotImplementedError    


def ensure_writeable(func: F) -> F:
    """A decorator that ensures that the wrapped method is only
    executed when the object is opened in writable mode

    Parameters
    ----------
    func : F
        The method that should only be executed if the method

    Returns
    -------
    F
        The same method with a check wrapped around it
    """        
    @functools.wraps(func)
    def wrapper(
            self: VectorStorage[Any, Any, Any], 
            *args: Any, **kwargs: Any) -> F:
        if not self.writeable:
            raise UnsupportedOperation(
                f"The object {self} is not writeable,"
                f" so the operation {func} is not supported.")
        return func(self, *args, **kwargs)
    return wrapper # type: ignore
