from abc import ABC, abstractmethod
from instancelib.environment.base import AbstractEnvironment
from typing import Any, Callable, Generic, Iterable, Sequence, TypeVar, Union
from uuid import UUID

from ..instances.base import Instance
from ..instances.text import TextInstance

from ..typehints import KT, VT


IT = TypeVar("IT", bound="Instance[Any, Any, Any, Any]")

class ChildGenerator(ABC, Generic[IT]):
    env: AbstractEnvironment[IT, Any, Any, Any, Any, Any]

    def register_child(self, parent: IT, child: IT) -> None:
        self.env.all_datapoints.add_child(parent, child)
class SinglePertubator(ABC, Generic[IT]):
    
    @abstractmethod
    def __call__(self, input: IT) -> IT:
        raise NotImplementedError

class MultiplePertubator(ABC, Generic[IT]):
    
    @abstractmethod
    def __call__(self, input: IT) -> Iterable[IT]:
        raise NotImplementedError

class TextPertubator(
        SinglePertubator[TextInstance[KT, VT]], 
        ChildGenerator[TextInstance[KT, VT]],
        Generic[KT, VT]):
    def __init__(self,
                 env: AbstractEnvironment[TextInstance[KT, VT], Union[KT, UUID], str, VT, str, Any],
                 pertubator:  Callable[[str], str]):
        self.env = env
        self.pertubator = pertubator

    def __call__(self, instance: TextInstance[KT, VT]) -> TextInstance[KT, VT]:
        input_text = instance.data
        pertubated_text = self.pertubator(input_text)
        new_instance = self.env.create(pertubated_text, None, pertubated_text)
        self.register_child(instance, new_instance)
        return new_instance


class TokenPertubator(SinglePertubator[TextInstance[KT, VT]], 
        ChildGenerator[TextInstance[KT, VT]],
        Generic[KT, VT]):
    def __init__(self,
                 env: AbstractEnvironment[TextInstance[KT, VT], Union[KT, UUID], str, VT, str, Any],
                 tokenizer: Callable[[str], Sequence[str]],
                 detokenizer: Callable[[Iterable[str]], str],
                 pertubator: Callable[[str], str]):
        self.env = env
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

        new_instance = self.env.create(data=new_data,
                                       vector=None,
                                       representation=new_data, 
                                       tokenized=new_tokenized)
        self.register_child(instance, new_instance)
        return new_instance
