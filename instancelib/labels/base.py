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

from abc import ABC, abstractmethod
from typing import Any, FrozenSet, Generic, Set, Union

from ..instances import Instance

from ..typehints import KT, LT



class LabelProvider(ABC, Generic[KT, LT]):
    @property
    @abstractmethod
    def labelset(self) -> FrozenSet[LT]:
        """Report all possible labels (example usage: for setting up a classifier)

        Returns
        -------
        Set[LT]
            Labels of type `LT`
        """
        raise NotImplementedError

    @abstractmethod
    def remove_labels(self, instance: Union[KT, Instance[KT, Any, Any, Any]], *labels: LT) -> None:
        """Remove the labels from this instance

        Parameters
        ----------
        instance : Union[KT, Instance]
            The instance
        *labels: LT
            The labels that should be removed from the instance
        """
        raise NotImplementedError

    @abstractmethod
    def set_labels(self, instance: Union[KT, Instance[KT, Any, Any, Any]], *labels: LT) -> None:
        """Annotate the instance with the labels listed in the parameters

        Parameters
        ----------
        instance : Union[KT, Instance]
            The instance
        *labels: LT
            The labels that should be associated with the instance
        """
        raise NotImplementedError

    @abstractmethod
    def get_labels(self, instance: Union[KT, Instance[KT, Any, Any, Any]]) -> FrozenSet[LT]:
        """Return the labels that are associated with the instance

        Parameters
        ----------
        instance : Union[KT, Instance]
            The instance

        Returns
        -------
        Set[LT]
            The labels that are associated with the instance
        """
        raise NotImplementedError

    @abstractmethod
    def get_instances_by_label(self, label: LT) -> FrozenSet[KT]:
        """Retrieve which instances are annotated with `label`

        Parameters
        ----------
        label : LT
            A Label

        Returns
        -------
        Set[Instance]
            The identifiers of the instance 
        """
        raise NotImplementedError

    @property
    def len_positive(self) -> int:
        docset: Set[KT] = set()
        for label in self.labelset:
            for instance in self.get_instances_by_label(label):
                docset.add(instance)
        return len(docset)

    def document_count(self, label: LT) -> int:
        return len(self.get_instances_by_label(label))
