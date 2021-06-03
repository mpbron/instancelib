from abc import ABC, abstractmethod
from typing import Callable, Generic, Iterable, Sequence, TypeVar, Union
from uuid import UUID

from ..instances.base import InstanceProvider
from ..instances.text import TextInstance

from ..typehints import KT, VT


IT = TypeVar("IT")

class AbstractPertubator(ABC, Generic[IT]):

    @abstractmethod
    def register_child(self, parent: IT, child: IT) -> None:
        raise NotImplementedError

    @abstractmethod
    def __call__(self, instance: IT) -> IT:
        raise NotImplementedError


class TextPertubator(AbstractPertubator[TextInstance[KT, VT]], Generic[KT, VT]):
    def __init__(self,
                 provider: InstanceProvider[TextInstance[KT, VT], Union[KT, UUID], str, VT, str],
                 pertubator:  Callable[[str], str]):
        self.provider = provider
        self.pertubator = pertubator

    def register_child(self,
                       parent: TextInstance[KT, VT],
                       child:  TextInstance[KT, VT]):
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
                 provider: InstanceProvider[TextInstance[KT, VT], Union[KT, UUID], str, VT, str],
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
