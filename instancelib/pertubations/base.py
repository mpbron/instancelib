from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, Iterable, Sequence, TypeVar, Union
from uuid import UUID

from ..instances.base import Instance
from ..instances.text import TextInstance, TextInstanceProvider
from ..typehints.typevars import KT, VT

InstanceType = TypeVar("InstanceType", bound="Instance[Any, Any, Any, Any]")


class AbstractPertubator(ABC, Generic[InstanceType]):

    @abstractmethod
    def register_child(self, parent: InstanceType, child: InstanceType) -> None:
        raise NotImplementedError

    @abstractmethod
    def __call__(self, instance: InstanceType) -> InstanceType:
        raise NotImplementedError


class TextPertubator(AbstractPertubator[TextInstance[KT, VT]], Generic[KT, VT]):
    def __init__(self,
                 provider: TextInstanceProvider[KT, VT],
                 pertubator:  Callable[[str], str]):
        self.provider = provider
        self.pertubator = pertubator

    def register_child(self,
                       parent: Instance[Union[KT, UUID], str, VT, str],
                       child: Instance[Union[KT, UUID], str, VT, str]):
        self.provider.add_child(parent, child)

    def __call__(self, instance: TextInstance[KT, VT]) -> TextInstance[KT, VT]:
        input_text = instance.data
        pertubated_text = self.pertubator(input_text)
        new_instance = self.provider.create(
            pertubated_text, None, pertubated_text)
        self.register_child(instance, new_instance)
        return new_instance


class TokenPertubator(TextPertubator[KT, VT], Generic[KT, VT]):
    def __init__(self,
                 provider: TextInstanceProvider[KT, VT],
                 tokenizer: Callable[[str], Sequence[str]],
                 detokenizer: Callable[[Iterable[str]], str],
                 pertubator: Callable[[str], str]):
        self.provider = provider
        self.tokenizer = tokenizer
        self.detokenizer = detokenizer
        self.pertubator = pertubator

    def __call__(self, instance: TextInstance[KT, VT]) -> TextInstance[KT, VT]:
        if not instance.tokenized:
            tokenized = self.tokenizer(instance.data)
            instance.tokenized = tokenized
        assert instance.tokenized
        new_tokenized = list(map(self.pertubator, instance.tokenized))
        new_data = self.detokenizer(new_tokenized)

        new_instance = self.provider.create(
            data=new_data,
            vector=None,
            representation=new_data)
        self.register_child(instance, new_instance)
        return new_instance
