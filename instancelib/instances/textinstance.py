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

from typing import Dict, Generic, Iterable, Optional, Sequence, Set, Union
from .memory import DataPoint, DataPointProvider
from ..typehints import KT, VT
from ..utils import to_key

class TextInstance(DataPoint[KT, str, VT, str], Generic[KT, VT]):
    def __init__(self, 
                 identifier: KT, 
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

class TextInstanceProvider(DataPointProvider[KT, str, VT, str], Generic[KT, VT]):

    def __init__(self, datapoints: Iterable[TextInstance[KT, VT]]) -> None:
        self.datapoints = {data.identifier: data for data in datapoints}
        self.children: Dict[KT, Set[KT]] = dict()
        self.parents: Dict[KT, KT] = dict()

    def add_child(self, 
                  parent: Union[KT, TextInstance[KT, VT]], 
                  child: Union[KT, TextInstance[KT, VT]]) -> None:
        parent_key: KT = to_key(parent)
        child_key: KT = to_key(parent)
        if parent_key in self and child_key in self:
            self.children.setdefault(parent_key, set()).add(child_key)
            self.parents[child_key] = parent_key
        else:
            raise KeyError("Either the parent or child does not exist in this Provider")

    def get_children(self, parent: Union[KT, TextInstance[KT, VT]]) -> Sequence[TextInstance[KT, VT]]:
        parent_key: KT = to_key(parent)
        if parent_key in self.children:
            children = [self.dictionary[child_key] for child_key in self.children[parent_key]]
            return children # type: ignore
        return []

    def get_parent(self, child: Union[KT, TextInstance[KT, VT]]) -> TextInstance[KT, VT]:
        child_key: KT = to_key(child)
        if child_key in self.parents:
            parent = self.dictionary[child_key]
            return parent # type: ignore
        raise KeyError(f"The instance with key {child_key} has no parent")

        
