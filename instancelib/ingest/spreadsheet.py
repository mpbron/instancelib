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

from ..environment import AbstractEnvironment
from ..environment.text import TextEnvironment
from ..instances.text import MemoryTextInstance
from ..utils.func import list_unzip3, single_or_collection


def identity_mapper(value: Any) -> Optional[str]:
    """Coerces any value to its string represenation

    Parameters
    ----------
    value : Any
        Any value that can be coerced into a string

    Returns
    -------
    Optional[str]
        The string representation of the value. If 
        coercion somehow failed, it will return None.
    """    
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

def extract_data_with_id(dataset_df: pd.DataFrame, 
                         id_col: str,
                         data_cols: Sequence[str], 
                         labelfunc: Callable[..., FrozenSet[str]]
               ) -> Tuple[List[Any], List[str], List[FrozenSet[str]]]:
    """Extract text data and labels from a dataframe

    Parameters
    ----------
    dataset_df : pd.DataFrame
        The dataset
    id_col: str
        The column where the identifier is stored
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
        for _, row in dataset_df.iterrows():
            identifier = row[id_col]
            data = " ".join([str(row[col]) for col in data_cols])
            labels = labelfunc(row)
            yield identifier, str(data), labels  # type: ignore
    indices, texts, labels_true = list_unzip3(yield_row_values())
    return indices, texts, labels_true  # type: ignore

def build_environment(df: pd.DataFrame, 
                      label_mapper: Callable[[Any], Optional[str]],
                      labels: Optional[Iterable[str]],
                      data_cols: Sequence[str],
                      label_cols: Sequence[str],
                     ) -> AbstractEnvironment[
                MemoryTextInstance[int, np.ndarray], 
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

def build_environment_with_id(df: pd.DataFrame, 
                      label_mapper: Callable[[Any], Optional[str]],
                      labels: Optional[Iterable[str]],
                      id_col: str,
                      data_cols: Sequence[str],
                      label_cols: Sequence[str],
                     ) -> AbstractEnvironment[
                MemoryTextInstance[Any, np.ndarray], 
                Union[Any, UUID], str, np.ndarray,str, str]:
    labelfunc = functools.partial(inv_transform_mapping, label_cols, label_mapper=label_mapper)
    indices, texts, true_labels = extract_data_with_id(df, id_col, data_cols, labelfunc)
    if labels is None:
        labels = frozenset(itertools.chain.from_iterable(true_labels))
    environment = TextEnvironment[int, np.ndarray, str].from_data(
        labels, 
        indices, texts, true_labels,
        [])
    return environment

def read_excel_dataset(path: "Union[str, PathLike[str]]", 
                       data_cols: Sequence[str], 
                       label_cols: Sequence[str], 
                       labels: Optional[Iterable[str]] = None,
                       label_mapper: Callable[[Any], Optional[str]] = identity_mapper
                       ) -> AbstractEnvironment[
                MemoryTextInstance[int, np.ndarray], 
                Union[int, UUID], str, np.ndarray, str, str]:
    """Read csv datasets that contain text data

    Parameters
    ----------
    path : Union[str, PathLike[str]]
        The path to the csv file
    data_cols : Sequence[str]
        The columns that contain the text data
    label_cols : Sequence[str]
        The columns that contain the columns
    labels : Optional[Iterable[str]], optional
        The set of labels that are possible.
        If None, the set will be inferred from data
        This parameter is by default None
    label_mapper : Callable[[Any], Optional[str]], optional
        A function that transferm labels into another representation
        This paramater is by default :func:`identity_mapper`, which just
        outputs its input.

    Returns
    -------
    AbstractEnvironment[TextInstance[int, np.ndarray], Union[int, UUID], str, np.ndarray, str, str]
        An environment that contains all the information from the CSV file
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
                MemoryTextInstance[int, np.ndarray], 
                Union[int, UUID], str, np.ndarray, str, str]:
    """Read Excel filse that contain text data

    Parameters
    ----------
    path : Union[str, PathLike[str]]
        The path to the Excel file
    data_cols : Sequence[str]
        The columns that contain the text data
    label_cols : Sequence[str]
        The columns that contain the columns
    labels : Optional[Iterable[str]], optional
        The set of labels that are possible.
        If None, the set will be inferred from data
        This parameter is by default None
    label_mapper : Callable[[Any], Optional[str]], optional
        A function that transferm labels into another representation
        This paramater is by default :func:`identity_mapper`, which just
        outputs its input.

    Returns
    -------
    AbstractEnvironment[TextInstance[int, np.ndarray], Union[int, UUID], str, np.ndarray, str, str]
        An environment that contains all the information from the Excel file
    """    
    df: pd.DataFrame = pd.read_csv(path) # type: ignore
    env = build_environment(df, label_mapper, labels, data_cols, label_cols)
    return env

def pandas_to_env(df: pd.DataFrame, 
                  data_cols: Union[str, Sequence[str]],
                  label_cols: Union[str, Sequence[str]],
                  labels: Optional[Iterable[str]] = None
                  ) -> AbstractEnvironment[
                    MemoryTextInstance[int, np.ndarray], 
                    Union[int, UUID], str, np.ndarray, str, str]:
    l_data_cols = single_or_collection(data_cols)
    l_label_cols = single_or_collection(label_cols)
    env = build_environment(df, identity_mapper, labels, l_data_cols, l_label_cols)
    return env

def pandas_to_env_with_id(df: pd.DataFrame, 
                  id_col: str,
                  data_cols: Union[str, Sequence[str]],
                  label_cols: Union[str, Sequence[str]],
                  labels: Optional[Iterable[str]] = None
                  ) -> AbstractEnvironment[MemoryTextInstance[Any, np.ndarray], 
                            Union[Any, UUID], str, np.ndarray, str, str]:
    l_data_cols = single_or_collection(data_cols)
    l_label_cols = single_or_collection(label_cols)
    env = build_environment_with_id(df, identity_mapper, labels, id_col,l_data_cols, l_label_cols)
    return env
