from abc import abstractmethod
from .base import InstanceProvider, Instance
from ..typehints import KT, DT, VT, RT
from typing import Any, Dict, TypeVar, Generic

IT = TypeVar("IT", bound="Instance[Any, Any, Any, Any]", covariant=True)


class ExternalProvider(InstanceProvider[IT, KT, DT, VT, RT], Generic[IT, KT, DT, VT, RT]):
    instance_cache: Dict[KT, IT]
    
    @abstractmethod
    def build_from_external(self, k: KT) -> IT:
        raise NotImplementedError

    @abstractmethod
    def update_external(self, ins: Instance[KT, DT, VT, RT]) -> None:
        raise NotImplementedError

    def clear_cache(self) -> None:
        self.instance_cache.clear()

    def __getitem__(self, k: KT) -> IT:
        if k in self:
            if k in self.instance_cache:
                instance = self.instance_cache[k]
                return instance
            instance = self.build_from_external(k)
            self.instance_cache[k] = instance
            return instance
        raise KeyError(f"Instance with key {k} is not present in this provider")

   

        

        