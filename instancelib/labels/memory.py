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
import collections

from typing import (Any, Callable, Dict, FrozenSet, Generic, Iterable, Iterator, Mapping, Optional,
                    Sequence, Set, Tuple, TypeVar, Union)

from ..instances import Instance
from ..typehints import KT, LT
from ..utils.func import list_unzip, union
from ..utils.to_key import to_key
from .base import LabelProvider

_T = TypeVar("_T")

class MemoryLabelProvider(LabelProvider[KT, LT], Generic[KT, LT]):
    """A Memory based implementation to test and benchmark AL algorithms
    """
    _labelset: FrozenSet[LT]
    _labeldict: Dict[KT, Set[LT]]
    _labeldict_inv: Dict[LT, Set[KT]]

    def __init__(self, 
            labelset: Iterable[LT], 
            labeldict: Dict[KT, Set[LT]], 
            labeldict_inv: Optional[Dict[LT, Set[KT]]] = None) -> None:
        self._labelset = frozenset(labelset)
        self._labeldict = labeldict
        if labeldict_inv is None:
            self._labeldict_inv = {label: set() for label in self._labelset}
            for key in self._labeldict.keys():
                for label in self._labeldict[key]:
                    self._labeldict_inv[label].add(key)
        else:
            self._labeldict_inv = labeldict_inv

    def __iter__(self) -> Iterator[KT]:
        return iter(self._labeldict)

    def __contains__(self, __o: object) -> bool:
        return to_key(__o) in self._labeldict

    def __len__(self) -> int:
        return len(self._labeldict)

    @classmethod
    def from_data(
            cls, 
            labelset: Iterable[LT], 
            indices: Sequence[KT], 
            labels: Sequence[Iterable[LT]]
        ) -> MemoryLabelProvider[KT, LT]:
        labelset = frozenset(labelset)
        labeldict = {
            idx: set(labellist) for (idx, labellist) in zip(indices, labels)
        }
        labeldict_inv: Dict[LT, Set[KT]] = {label: set() for label in labelset}
        # Store all instances in a Dictionary<LT, Set[ID]>
        for key, labellist in labeldict.items():
            for label in labellist:
                labeldict_inv[label].add(key)
        return cls(labelset, labeldict, labeldict_inv)

    @classmethod
    def from_provider(cls, provider: LabelProvider[KT, LT], subset: Iterable[KT] = list()) -> MemoryLabelProvider[KT, LT]:
        instances = frozenset(subset) if subset else frozenset(provider.keys())
        labelset = provider.labelset
        labeldict_inv = {label: set(provider.get_instances_by_label(label).intersection(instances)) for label in labelset}
        labeldict: Dict[KT, Set[LT]]= {}
        for label, key_list in labeldict_inv.items():
            for key in key_list:
                labeldict.setdefault(key, set()).add(label)
        return cls(labelset, labeldict, labeldict_inv)

    @classmethod
    def from_tuples(cls, predictions: Sequence[Tuple[KT, FrozenSet[LT]]]) -> MemoryLabelProvider[KT, LT]:
        _, labels = list_unzip(predictions)
        labelset = union(*labels)
        labeldict = {key: set(labeling) for (key, labeling) in predictions}
        provider = cls(labelset, labeldict, None)
        return  provider

    @property
    def labelset(self) -> FrozenSet[LT]:
        return self._labelset

    def remove_labels(self, instance: Union[KT, Instance[KT, Any, Any, Any]], *labels: LT):
        key = to_key(instance)
        if key not in self._labeldict:
            raise KeyError("Key {} is not found".format(key))
        for label in labels:
            self._labeldict[key].discard(label)
            self._labeldict_inv[label].discard(key)

    def set_labels(self, instance: Union[KT, Instance[KT, Any, Any, Any]], *labels: LT):
        key = to_key(instance)
        for label in labels:
            self._labeldict.setdefault(key, set()).add(label)
            self._labeldict_inv.setdefault(label, set()).add(key)

    def get_labels(self, instance: Union[KT, Instance[KT, Any, Any, Any]]) -> FrozenSet[LT]:
        key = to_key(instance)
        if key in self:
            return frozenset(self._labeldict[key])
        return frozenset()

    def get_instances_by_label(self, label: LT) -> FrozenSet[KT]:
        return frozenset(self._labeldict_inv.setdefault(label, set()))

    def document_count(self, label: LT) -> int:
        return len(self.get_instances_by_label(label))
            
    @classmethod
    def rename_labels(cls, 
                      provider: LabelProvider[KT, _T], 
                      mapping: Union[Mapping[_T, LT], Callable[[_T], LT]]
                      ) -> MemoryLabelProvider[KT, LT]:
        def translate_from_map(labelmap: Mapping[_T, LT]) -> Callable[[_T], LT]:
            def inside(value: _T) -> LT:
                if value in labelmap:
                    return labelmap[value]
                return value # type: ignore
            return inside
        mapper = translate_from_map(mapping) if isinstance(mapping, collections.Mapping) else mapping
        labeldict = {key: {mapper(old_label) for old_label in old_labels} for key, old_labels in provider.items()}
        labelset = frozenset([mapper(lbl) for lbl in provider.labelset])
        provider = cls(labelset, labeldict, None) # type: ignore
        return provider # type: ignore


