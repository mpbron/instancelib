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

import itertools
from typing import Any, Generic, Iterable, Optional, Sequence, Union
from uuid import UUID, uuid4

from ..typehints import KT, VT
from .memory import AbstractMemoryProvider, DataPoint


class TextInstance(DataPoint[Union[KT, UUID], str, VT, str], Generic[KT, VT]):
    def __init__(self, 
                 identifier: Union[KT, UUID], 
                 data: str, 
                 vector: Optional[VT], 
                 representation: Optional[str] = None, 
                 tokenized: Optional[Sequence[str]] = None) -> None:
        
        representation = data if representation is None else representation
        super().__init__(identifier, data, vector, representation)
        self._tokenized = tokenized
    
    @property
    def tokenized(self) -> Optional[Sequence[str]]:
        return self._tokenized
    
    @tokenized.setter
    def tokenized(self, value: Sequence[str]) -> None:
        self._tokenized = value

class TextInstanceProvider(AbstractMemoryProvider[TextInstance[KT, VT], Union[KT, UUID], str, VT, str], Generic[KT, VT]):

    def __init__(self, 
                 datapoints: Iterable[TextInstance[KT, VT]],
                    ) -> None:
        self.dictionary = {data.identifier: data for data in datapoints}
        self.children = dict()
        self.parents = dict()

    @classmethod
    def from_data_and_indices(cls, # type: ignore
                              indices: Sequence[KT],
                              raw_data: Sequence[str],
                              vectors: Optional[Sequence[Optional[VT]]] = None) -> TextInstanceProvider[KT, VT]:
        if vectors is None or len(vectors) != len(indices):
            vectors = [None] * len(indices)
        datapoints = itertools.starmap(
            TextInstance[KT,VT], zip(indices, raw_data, vectors, raw_data))
        return cls(datapoints)

    @classmethod
    def from_data(cls, raw_data: Sequence[str]) -> TextInstanceProvider[KT, VT]:
        indices = range(len(raw_data))
        vectors = [None] * len(raw_data)
        datapoints = itertools.starmap(TextInstance[KT, VT], zip(indices, raw_data, vectors, raw_data))
        return cls(datapoints)

    def create(self, *args: Any, **kwargs: Any):  # type: ignore
        new_key = uuid4()
        new_instance = TextInstance[KT, VT](new_key, *args, **kwargs)
        self.add(new_instance)
        return new_instance

