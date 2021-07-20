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

import numpy as np
from typing import Any, Generic, Optional, Sequence, Union
from uuid import UUID, uuid4

from ..typehints import KT, VT
from .memory import AbstractMemoryProvider, DataPoint
from .base import Instance


class TextInstance(Instance[KT, str, VT, str], ABC, Generic[KT, VT]):
    
    @property
    @abstractmethod
    def map_to_original(self) -> Optional[np.ndarray]:
        raise NotImplementedError

    @map_to_original.setter
    @abstractmethod
    def map_to_original(self, value: Optional[np.ndarray]) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def split_marker(self) -> Optional[Any]:
        raise NotImplementedError

    @split_marker.setter
    @abstractmethod
    def split_marker(self, value: Any) -> None:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def tokenized(self) -> Optional[Sequence[str]]:
        raise NotImplementedError
    
    @tokenized.setter
    @abstractmethod
    def tokenized(self, value: Sequence[str]) -> None:
        raise NotImplementedError


class MemoryTextInstance(DataPoint[Union[KT, UUID], str, VT, str], TextInstance[Union[KT, UUID], VT], Generic[KT, VT]):
   
    def __init__(self, 
                 identifier: Union[KT, UUID], 
                 data: str, 
                 vector: Optional[VT], 
                 representation: Optional[str] = None, 
                 tokenized: Optional[Sequence[str]] = None,
                 map_to_original: Optional[np.ndarray] = None,
                 split_marker: Optional[Any] = None) -> None:
        representation = data if representation is None else representation
        super().__init__(identifier, data, vector, representation)
        self._tokenized = tokenized
        self._map_to_original = map_to_original
        self._split_marker = split_marker

    @property
    def map_to_original(self) -> Optional[np.ndarray]:
        return self._map_to_original

    @map_to_original.setter
    def map_to_original(self, value: Optional[np.ndarray]) -> None:
        self._map_to_original = value

    @property
    def split_marker(self) -> Optional[Any]:
        return self._split_marker 

    @split_marker.setter
    def split_marker(self, value: Any):
        self._split_marker = value
    
    @property
    def tokenized(self) -> Optional[Sequence[str]]:
        return self._tokenized
    
    @tokenized.setter
    def tokenized(self, value: Sequence[str]) -> None:
        self._tokenized = value

        

class TextInstanceProvider(AbstractMemoryProvider[MemoryTextInstance[KT, VT], Union[KT, UUID], str, VT, str], Generic[KT, VT]):

    def create(self, *args: Any, **kwargs: Any):
        new_key = uuid4()
        new_instance = MemoryTextInstance[KT, VT](new_key, *args, **kwargs)
        self.add(new_instance)
        return new_instance

    @staticmethod
    def construct(*args: Any, **kwargs: Any) -> MemoryTextInstance[KT, VT]:
        return MemoryTextInstance[KT, VT](*args, **kwargs)
