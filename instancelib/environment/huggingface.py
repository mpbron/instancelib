# from __future__ import annotations

# from typing import Generic, Union
# from uuid import UUID

# import numpy as np

# from ..instances.dataset import ReadOnlyProvider
# from ..instances.huggingface import HuggingFaceDataset
# from ..instances.text import MemoryTextInstance
# from ..typehints.typevars import KT, LT
# from .memory import MemoryEnvironment

# def build(identifier: KT, text: str) -> MemoryTextInstance[KT, npt.NDArray[Any]]

# class HuggingFaceTextEnvironment(MemoryEnvironment[MemoryTextInstance[KT, npt.NDArray[Any]], Union[KT, UUID], str, npt.NDArray[Any], str, LT], Generic[KT, LT]):
#     @classmethod
#     def from_wrapped(cls, hds: HuggingFaceDataset[KT, str]) -> HuggingFaceTextEnvironment[KT, LT]:
#         provider = ReadOnlyProvider[(hds, lambda kt, dt: )
