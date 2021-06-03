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

import functools
import itertools
from os import PathLike
from typing import (Any, Callable, FrozenSet, Iterable, Iterator, List,
                    Optional, Sequence, Tuple, Union)
from uuid import UUID

import numpy as np
import pandas as pd

from ..environment.base import AbstractEnvironment
from ..environment.text import TextEnvironment
from ..instances.text import TextInstance
from ..utils.func import list_unzip3


def identity_mapper(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return value
    coerced = str(value)
    if not coerced:
        return None
    return coerced


def inv_transform_mapping(columns: Sequence[str],
                          row: "pd.Series[str]", 
                          label_mapper: Callable[[Any], Optional[str]] = identity_mapper,
                          ) -> FrozenSet[str]:
    """Convert the numeric coded label in column `column_name` in row `row`
    to a string according to the mapping in `label_mapping`.

    Parameters
    ----------
    column_name : str
        The column in which the labels are stored
    row : pd.Series
        A row from a Pandas DataFrame
    label_mapper : Callable[[Any], str], optional
        A mapping from values to strings, by default `identity_mapper`,
        a function that coerces values to strings

    Returns
    -------
    FrozenSet[str]
        A set of labels that belong to the row
    """
    def read_columns() -> Iterator[str]:
        for column in columns:
            coded_label = row[column]
            decoded_label = label_mapper(coded_label)
            if decoded_label is not None:
                yield decoded_label
    return frozenset(read_columns())

def extract_data(dataset_df: pd.DataFrame, 
               data_cols: Sequence[str], 
               labelfunc: Callable[..., FrozenSet[str]]
               ) -> Tuple[List[int], List[str], List[FrozenSet[str]]]:
    """Extract text data and labels from a dataframe

    Parameters
    ----------
    dataset_df : pd.DataFrame
        The dataset
    data_cols : List[str]
        The cols in which the text is stored
    labelfunc : Callable[..., FrozenSet[str]]
        A function that maps rows to sets of labels

    Returns
    -------
    Tuple[List[int], List[str], List[FrozenSet[str]]]
        [description]
    """    
    def yield_row_values():
        for i, row in dataset_df.iterrows():
            data = " ".join([str(row[col]) for col in data_cols])
            labels = labelfunc(row)
            yield int(i), str(data), labels  # type: ignore
    indices, texts, labels_true = list_unzip3(yield_row_values())
    return indices, texts, labels_true  # type: ignore

def build_environment(df: pd.DataFrame, 
                      label_mapper: Callable[[Any], Optional[str]],
                      labels: Optional[Iterable[str]],
                      data_cols: Sequence[str],
                      label_cols: Sequence[str],
                     ) -> AbstractEnvironment[
                TextInstance[int, np.ndarray], 
                Union[int, UUID], str, np.ndarray, str, str]:
    """Build an environment from a data frame

    Parameters
    ----------
    df : pd.DataFrame
        A data frame that contains all texts and labels
    label_mapping : Mapping[int, str]
        A mapping from indices to label strings
    data_cols : Sequence[str]
        A sequence of columns that contain the texts
    label_col : str
        The name of the column that contains the label data

    Returns
    -------
    MemoryEnvironment[int, str, np.ndarray, str]
        A MemoryEnvironment that contains the  
    """    
    labelfunc = functools.partial(inv_transform_mapping, label_cols, label_mapper=label_mapper)
    indices, texts, true_labels = extract_data(df, data_cols, labelfunc)
    if labels is None:
        labels = frozenset(itertools.chain.from_iterable(true_labels))
    environment = TextEnvironment[int, np.ndarray, str].from_data(
        labels, 
        indices, texts, true_labels,
        [])
    return environment

Environment = AbstractEnvironment[
                TextInstance[int, np.ndarray], 
                Union[int, UUID], str, np.ndarray,str, str]

def read_excel_dataset(path: "Union[str, PathLike[str]]", 
                       data_cols: Sequence[str], 
                       label_cols: Sequence[str], 
                       labels: Optional[Iterable[str]] = None,
                       label_mapper: Callable[[Any], Optional[str]] = identity_mapper
                       ) -> AbstractEnvironment[
                TextInstance[int, np.ndarray], 
                Union[int, UUID], str, np.ndarray, str, str]:
    """Convert a Excel Dataset

    Parameters
    ----------
    path : PathLike
        The path to the Excel file

    Returns
    -------
    MemoryEnvironment[int, str, np.ndarray, str]
        A MemoryEnvironment. The labels that 
    """    
    df: pd.DataFrame = pd.read_excel(path) # type: ignore
    env = build_environment(df, label_mapper, labels, data_cols, label_cols)
    return env

def read_csv_dataset(path: "Union[str, PathLike[str]]", 
                       data_cols: Sequence[str], 
                       label_cols: Sequence[str], 
                       labels: Optional[Iterable[str]] = None,
                       label_mapper: Callable[[Any], Optional[str]] = identity_mapper
                       ) -> AbstractEnvironment[
                TextInstance[int, np.ndarray], 
                Union[int, UUID], str, np.ndarray, str, str]:
    """Convert a Excel Dataset

    Parameters
    ----------
    path : PathLike
        The path to the CSV file

    Returns
    -------
    MemoryEnvironment[int, str, np.ndarray, str]
        A MemoryEnvironment. The labels that 
    """    
    df: pd.DataFrame = pd.read_csv(path) # type: ignore
    env = build_environment(df, label_mapper, labels, data_cols, label_cols)
    return env
