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
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)
from uuid import UUID

import numpy.typing as npt
import pandas as pd

from instancelib.environment.memory import MemoryEnvironment
from instancelib.labels.memory import MemoryLabelProvider

from ..environment import AbstractEnvironment
from ..environment.text import TextEnvironment
from ..instances.base import Instance, InstanceProvider
from ..instances.text import MemoryTextInstance, TextInstanceProvider
from ..labels.base import LabelProvider
from ..typehints import DT, KT, LT, RT, VT
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


def inv_transform_mapping(
    columns: Sequence[str],
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


def extract_data(
    dataset_df: pd.DataFrame,
    data_cols: Sequence[str],
    labelfunc: Callable[..., FrozenSet[str]],
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


def extract_data_with_id(
    dataset_df: pd.DataFrame,
    id_col: str,
    data_cols: Sequence[str],
    labelfunc: Callable[..., FrozenSet[str]],
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


def build_environment(
    df: pd.DataFrame,
    label_mapper: Callable[[Any], Optional[str]],
    labels: Optional[Iterable[str]],
    data_cols: Sequence[str],
    label_cols: Sequence[str],
) -> AbstractEnvironment[
    MemoryTextInstance[int, npt.NDArray[Any]],
    Union[int, UUID],
    str,
    npt.NDArray[Any],
    str,
    str,
]:
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
    MemoryEnvironment[int, str, npt.NDArray[Any], str]
        A MemoryEnvironment that contains the
    """
    labelfunc = functools.partial(
        inv_transform_mapping, label_cols, label_mapper=label_mapper
    )
    indices, texts, true_labels = extract_data(df, data_cols, labelfunc)
    if labels is None:
        labels = frozenset(itertools.chain.from_iterable(true_labels))
    environment = TextEnvironment[int, npt.NDArray[Any], str].from_data(
        labels, indices, texts, true_labels, []
    )
    return environment


def build_environment_with_id(
    df: pd.DataFrame,
    label_mapper: Callable[[Any], Optional[str]],
    labels: Optional[Iterable[str]],
    id_col: str,
    data_cols: Sequence[str],
    label_cols: Sequence[str],
) -> AbstractEnvironment[
    MemoryTextInstance[Any, npt.NDArray[Any]],
    Union[Any, UUID],
    str,
    npt.NDArray[Any],
    str,
    str,
]:
    labelfunc = functools.partial(
        inv_transform_mapping, label_cols, label_mapper=label_mapper
    )
    indices, texts, true_labels = extract_data_with_id(
        df, id_col, data_cols, labelfunc
    )
    if labels is None:
        labels = frozenset(itertools.chain.from_iterable(true_labels))
    environment = TextEnvironment[int, npt.NDArray[Any], str].from_data(
        labels, indices, texts, true_labels, []
    )
    return environment


def read_excel_dataset(
    path: "Union[str, PathLike[str]]",
    data_cols: Sequence[str],
    label_cols: Sequence[str],
    labels: Optional[Iterable[str]] = None,
    label_mapper: Callable[[Any], Optional[str]] = identity_mapper,
) -> AbstractEnvironment[
    MemoryTextInstance[int, npt.NDArray[Any]],
    Union[int, UUID],
    str,
    npt.NDArray[Any],
    str,
    str,
]:
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
    AbstractEnvironment[TextInstance[int, npt.NDArray[Any]], Union[int, UUID], str, npt.NDArray[Any], str, str]
        An environment that contains all the information from the CSV file
    """
    df: pd.DataFrame = pd.read_excel(path)  # type: ignore
    env = build_environment(df, label_mapper, labels, data_cols, label_cols)
    return env


def read_csv_dataset(
    path: "Union[str, PathLike[str]]",
    data_cols: Sequence[str],
    label_cols: Sequence[str],
    labels: Optional[Iterable[str]] = None,
    label_mapper: Callable[[Any], Optional[str]] = identity_mapper,
) -> AbstractEnvironment[
    MemoryTextInstance[int, npt.NDArray[Any]],
    Union[int, UUID],
    str,
    npt.NDArray[Any],
    str,
    str,
]:
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
    AbstractEnvironment[TextInstance[int, npt.NDArray[Any]], Union[int, UUID], str, npt.NDArray[Any], str, str]
        An environment that contains all the information from the Excel file
    """
    df: pd.DataFrame = pd.read_csv(path)  # type: ignore
    env = build_environment(df, label_mapper, labels, data_cols, label_cols)
    return env


def pandas_to_env(
    df: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    data_cols: Union[str, Sequence[str]],
    label_cols: Union[str, Sequence[str]],
    labels: Optional[Iterable[str]] = None,
) -> AbstractEnvironment[
    MemoryTextInstance[Any, npt.NDArray[Any]],
    Union[Any, UUID],
    str,
    npt.NDArray[Any],
    str,
    str,
]:
    l_data_cols = single_or_collection(data_cols)
    l_label_cols = single_or_collection(label_cols)
    if isinstance(df, dict):
        env = build_from_multiple_dfs(
            df, identity_mapper, labels, l_data_cols, l_label_cols
        )
    else:
        env = build_environment(
            df, identity_mapper, labels, l_data_cols, l_label_cols
        )
    return env


def pandas_to_env_with_id(
    df: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    id_col: str,
    data_cols: Union[str, Sequence[str]],
    label_cols: Union[str, Sequence[str]],
    labels: Optional[Iterable[str]] = None,
) -> AbstractEnvironment[
    MemoryTextInstance[Any, npt.NDArray[Any]],
    Union[Any, UUID],
    str,
    npt.NDArray[Any],
    str,
    str,
]:
    l_data_cols = single_or_collection(data_cols)
    l_label_cols = single_or_collection(label_cols)
    if isinstance(df, dict):
        env = build_from_multiple_dfs_with_ids(
            df, identity_mapper, labels, id_col, l_data_cols, l_label_cols
        )
    else:
        env = build_environment_with_id(
            df, identity_mapper, labels, id_col, l_data_cols, l_label_cols
        )
    return env


IT = TypeVar("IT", bound="Instance[Any, Any, Any, Any]")


def instance_extractor(
    df: pd.DataFrame,
    id_extractor: Callable[[pd.Series, Any], KT],
    data_extractor: Callable[[pd.Series], DT],
    vector_extractor: Callable[[pd.Series], VT],
    repr_extractor: Callable[[pd.Series], RT],
    label_extractor: Callable[[pd.Series], FrozenSet[LT]],
    builder: Callable[[KT, DT, VT, RT, pd.Series, Any], IT],
) -> Iterator[Tuple[KT, IT, FrozenSet[LT]]]:
    def row_func(row: pd.Series, idx: Any) -> Tuple[KT, IT, FrozenSet[LT]]:
        identifier = id_extractor(row, idx)
        data = data_extractor(row)
        vector = vector_extractor(row)
        representation = repr_extractor(row)
        labels = label_extractor(row)
        instance = builder(identifier, data, vector, representation, row, idx)
        return identifier, instance, labels

    for idx, row in df.iterrows():
        yield row_func(row, idx)


def to_dicts(
    triples: Iterator[Tuple[KT, IT, FrozenSet[LT]]]
) -> Tuple[Mapping[KT, IT], Mapping[KT, FrozenSet[LT]]]:
    keys, instances, labels = list_unzip3(triples)
    instance_dict = dict(zip(keys, instances))
    label_dict = dict(zip(keys, labels))
    return instance_dict, label_dict


def to_environment(
    prov_builder: Callable[
        [Mapping[KT, IT]], InstanceProvider[IT, KT, DT, VT, RT]
    ],
    labelprov_builder: Callable[
        [Mapping[KT, FrozenSet[LT]]], LabelProvider[KT, LT]
    ],
    dictionaries: Tuple[Mapping[KT, IT], Mapping[KT, FrozenSet[LT]]],
) -> AbstractEnvironment[IT, KT, DT, VT, RT, LT]:
    instances, labels = dictionaries
    ins_provider = prov_builder(instances)
    lbl_provider = labelprov_builder(labels)
    return MemoryEnvironment(ins_provider, lbl_provider)


def build_from_multiple_dfs(
    df_dict: Dict[str, pd.DataFrame],
    label_mapper: Callable[[Any], Optional[str]],
    labels: Optional[Iterable[str]],
    data_cols: Sequence[str],
    label_cols: Sequence[str],
) -> AbstractEnvironment[
    MemoryTextInstance[str, npt.NDArray[Any]],
    Union[str, UUID],
    str,
    npt.NDArray[Any],
    str,
    str,
]:
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
    MemoryEnvironment[int, str, npt.NDArray[Any], str]
        A MemoryEnvironment that contains the
    """
    labelfunc = functools.partial(
        inv_transform_mapping, label_cols, label_mapper=label_mapper
    )
    indices_table: Dict[str, List[str]] = dict()
    indices: List[str] = list()
    texts: List[str] = list()
    true_labels: List[FrozenSet[str]] = list()

    for df_key, df in df_dict.items():
        idxs, df_texts, df_true_labels = extract_data(df, data_cols, labelfunc)
        indices_table[df_key] = [f"{df_key}_{idx}" for idx in idxs]
        indices = indices + indices_table[df_key]
        texts = texts + df_texts
        true_labels = true_labels + df_true_labels
    if labels is None:
        labels = frozenset(itertools.chain.from_iterable(true_labels))
    environment = TextEnvironment[str, npt.NDArray[Any], str].from_data(
        labels, indices, texts, true_labels, []
    )
    for key, split_indices in indices_table.items():
        environment[key] = environment.create_bucket(split_indices)
    return environment


def build_from_multiple_dfs_with_ids(
    df_dict: Dict[str, pd.DataFrame],
    label_mapper: Callable[[Any], Optional[str]],
    labels: Optional[Iterable[str]],
    id_col: str,
    data_cols: Sequence[str],
    label_cols: Sequence[str],
) -> AbstractEnvironment[
    MemoryTextInstance[str, npt.NDArray[Any]],
    Union[str, UUID],
    str,
    npt.NDArray[Any],
    str,
    str,
]:
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
    MemoryEnvironment[int, str, npt.NDArray[Any], str]
        A MemoryEnvironment that contains the
    """
    labelfunc = functools.partial(
        inv_transform_mapping, label_cols, label_mapper=label_mapper
    )
    indices_table: Dict[str, List[str]] = dict()
    indices: List[str] = list()
    texts: List[str] = list()
    true_labels: List[FrozenSet[str]] = list()

    for df_key, df in df_dict.items():
        indices_table[df_key], df_texts, df_true_labels = extract_data_with_id(
            df, id_col, data_cols, labelfunc
        )
        indices = indices + indices_table[df_key]
        texts = texts + df_texts
        true_labels = true_labels + df_true_labels
    if labels is None:
        labels = frozenset(itertools.chain.from_iterable(true_labels))
    environment = TextEnvironment[str, npt.NDArray[Any], str].from_data(
        labels, indices, texts, true_labels, []
    )
    for key, split_indices in indices_table.items():
        environment[key] = environment.create_bucket(split_indices)
    return environment


def one_hot_encoded_extractor(
    *cols: str,
) -> Callable[[pd.Series], FrozenSet[str]]:
    def callable(row: pd.Series) -> FrozenSet[str]:
        lbls = frozenset([col for col in cols if row[col]])
        return lbls

    return callable


def text_concatenation(*cols: str) -> Callable[[pd.Series], str]:
    def callable(row: pd.Series) -> str:
        text = " ".join([col for col in cols if row[col]])
        return text

    return callable


def no_vector() -> Callable[[pd.Series], Optional[npt.NDArray[Any]]]:
    def callable(row: pd.Series) -> Optional[npt.NDArray[Any]]:
        return None

    return callable


def id_col(col: str) -> Callable[[pd.Series, Any], Any]:
    def callable(row: pd.Series, idx: Any) -> Any:
        identifier = row[col]
        return identifier

    return callable


def id_index() -> Callable[[pd.Series, Any], Any]:
    def callable(row: pd.Series, idx: Any) -> Any:
        return idx

    return callable


def id_index_prefix(prefix: str) -> Callable[[pd.Series, Any], str]:
    def callable(row: pd.Series, idx: Any) -> str:
        return f"{prefix}_{idx}"

    return callable


def text_builder(
    identifier: KT,
    data: str,
    vector: VT,
    representation: str,
    row: pd.Series,
    idx: Any,
) -> MemoryTextInstance[KT, VT]:
    return MemoryTextInstance(identifier, data, vector, representation)


def text_from_pandas_multilabel(
    df_dict: Dict[str, pd.DataFrame],
    text_cols: Sequence[str],
    label_cols: Sequence[str],
    labelset: FrozenSet[str],
):
    def instances() -> Iterator[
        Tuple[str, MemoryTextInstance[str, Any], FrozenSet[str]]
    ]:
        for name, df in df_dict.items():
            df_instances = instance_extractor(
                df,
                id_index_prefix(name),
                text_concatenation(*text_cols),
                no_vector(),
                text_concatenation(*text_cols),
                one_hot_encoded_extractor(*label_cols),
                text_builder,
            )
            yield from df_instances

    dicts = to_dicts(instances())
    pass
