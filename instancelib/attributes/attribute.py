from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (FrozenSet, Generic, Iterable, Mapping, Optional, Sequence,
                    TypeVar)

from instancelib.typehints import LT
from instancelib.utils.func import powerset, union

NT = TypeVar("NT")


class BaseAttribute(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the attribute

        Returns
        -------
        str
            The name of the attribute
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def description(self) -> str:
        """A human readable description of the attribute

        Returns
        -------
        str
            A human readable description
        """
        raise NotImplementedError


class NamedAttribute(BaseAttribute):
    _name: str
    _description: str

    def __init__(self,
                 name: str,
                 description: Optional[str] = None) -> None:
        self._name = name
        self._description = "" if description is None else description

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description


class Choices(ABC, Generic[LT]):
    @property
    def options(self) -> FrozenSet[LT]:
        """The possible valid choices for this attribute

        Returns
        -------
        FrozenSet[LT]
            A set with all possible attributes
        """
        raise NotImplementedError

    def alternatives(self, *option: LT) -> FrozenSet[LT]:
        """Give back other options for this attribute than 
        the one in the parameters

        Parameters
        ----------
        option : LT
            One or multiple options to be excluded

        Returns
        -------
        FrozenSet[LT]
            The alternatives without the given option(s)
        """
        raise NotImplementedError


class ChoiceAttribute(NamedAttribute, Choices[LT], Generic[LT]):

    def __init__(self,
                 name: str,
                 options: Iterable[LT],
                 description: Optional[str] = None,
                 ) -> None:
        super().__init__(name, description=description)
        self._options = frozenset(options)

    @property
    def options(self) -> FrozenSet[LT]:
        return self._options

    def alternatives(self, *option: LT) -> FrozenSet[LT]:
        return self.options.difference(option)


class BinaryChoiceAttribute(ChoiceAttribute[LT], Generic[LT]):

    def __init__(self, name: str, positive: LT, negative: LT, description: Optional[str] = None) -> None:
        super().__init__(name, [positive, negative], description=description)
        self._positive = positive
        self._negative = negative

    @property
    def positive(self) -> LT:
        """The positive choice

        Returns
        -------
        LT
            The positive label
        """
        return self._positive

    @property
    def negative(self) -> LT:
        """The negative choice

        Returns
        -------
        LT
            The negative label
        """
        return self._negative


class MultiLabelAttribute(ChoiceAttribute[FrozenSet[LT]], Generic[LT]):

    def __init__(self, name: str, options: Iterable[LT], description: Optional[str] = None) -> None:
        powerset_options = powerset(options)
        super().__init__(name, powerset_options, description=description)


class LabelTree(Choices[LT], Generic[LT]):
    _label: Optional[LT]
    children: Sequence[LabelTree[LT]]

    def __init__(self, *children: LabelTree[LT]):
        self._label = None
        self.children = list(children)

    @property
    def label(self) -> Optional[LT]:
        return self._label

    def __str__(self) -> str:
        children = ", ".join(map(str, self.children))
        label = ""
        if not self.is_root:
            label = f"{self.label}"
        if children and self.is_root:
            return f"[{children}]"
        if children:
            return f"{label}: [{children}]"
        return label

    def __repr__(self) -> str:
        return str(self)

    @property
    def options(self) -> FrozenSet[LT]:
        child_labels = (
            child.label for child in self.children if child.label is not None)
        return frozenset(child_labels)

    def alternatives(self, *option: LT) -> FrozenSet[LT]:
        if all(map(lambda x: x in self.options, option)):
            return self.options.difference(option)
        if self.children:
            return union(*(child.alternatives(*option) for child in self.children))
        return frozenset()

    @property
    def all_options(self) -> FrozenSet[LT]:
        current_options = self.options
        deeper_options = union(*(child.all_options for child in self.children))
        return union(current_options, deeper_options)

    @property
    def leaves(self) -> FrozenSet[LT]:
        if self.children:
            return union(*(child.leaves for child in self.children))
        if self.label is None:
            return frozenset()
        return frozenset([self.label])

    def is_parent_of(self, option: LT) -> bool:
        return option in self.options

    def is_ancestor_of(self, option: LT) -> bool:
        return option in self.all_options

    def parent(self, option: LT) -> Optional[LT]:
        if option in self.options:
            return self.label
        if self.is_ancestor_of(option):
            for child in self.children:
                result = child.parent(option)
                if result is not None:
                    return result
        return None

    @property
    def is_root(self) -> bool:
        return self._label is None

    def children_of(self, option: LT) -> FrozenSet[LT]:
        if self.label == option:
            return self.options
        if self.is_ancestor_of(option):
            child_results = (child.children_of(option)
                             for child in self.children)
            return union(*child_results)
        return frozenset()

    def ancestors(self, option: LT) -> FrozenSet[LT]:
        parent = self.parent(option)
        if parent is None:
            return frozenset()
        return union(frozenset([parent]), self.ancestors(parent))


class LabelNode(LabelTree[LT], Generic[LT]):
    def __init__(self, label: LT, *children: LabelTree[LT]):
        super().__init__(*children)
        self._label = label


class HierarchyAttribute(NamedAttribute, LabelTree[LT], Generic[LT]):
    def __init__(self, name: str, *children: LabelTree[LT],  description: Optional[str] = None) -> None:
        super().__init__(name, description=description)
        self.children = list(children)
        self._label = None

    @classmethod
    def from_dict(cls, name: str, dictionary: Mapping[LT, Sequence[LT]], root: LT) -> HierarchyAttribute[LT]:
        def sub_func(node: LT) -> LabelTree[LT]:
            if node in dictionary:
                children = dictionary[node]
                return LabelNode(node, *map(sub_func, children))
            return LabelNode(node)
        children = dictionary[root]
        return cls(name, *map(sub_func, children))
