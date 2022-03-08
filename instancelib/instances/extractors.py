from abc import ABC
from typing import Any, Generic, Mapping, TypeVar


_T = TypeVar("_T")

class DataExtractor(ABC, Generic[_T]):
    def __call__(self, row: Mapping[str, Any]) -> _T:
        raise NotImplementedError

class IdentityExtractor(DataExtractor[Mapping[str, Any]]):
    def __call__(self, row: Mapping[str, Any]) -> Mapping[str, Any]:
        return row

class ColumnExtractor(DataExtractor[_T], Generic[_T]):
    def __init__(self, column: str):
        self.column = column

    def __call__(self, row: Mapping[str, Any]) -> _T:
        return row[self.column]

class ConcatenationExtractor(DataExtractor[str]):
    def __init__(self, *columns: str, seperator: str = " "):
        self.columns = list(columns)
        self.seperator = seperator

    def __call__(self, row: Mapping[str, Any]) -> str:
        return self.seperator.join((row[col] for col in self.columns))

class SubsetExtractor(IdentityExtractor):
    def __init__(self, *columns: str):
        self.columns = tuple(columns)

    def __call__(self, row: Mapping[str, Any]) -> Mapping[str, Any]:
        return {col: row[col] for col in self.columns}