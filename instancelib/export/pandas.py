from typing import Any, Callable, Iterator, Mapping, Optional, Sequence, Tuple, TypeVar

import pandas as pd

from instancelib.utils.func import flatten_dicts

from ..instances.base import (
    Instance,
    InstanceProvider,
    default_instance_viewer,
)
from ..labels.base import LabelProvider, default_label_viewer
from ..typehints.typevars import DT, KT, LT, RT, VT

IT = TypeVar("IT", bound="Instance[Any, Any, Any, Any]", covariant=True)


def to_pandas(
    provider: InstanceProvider[IT, KT, DT, VT, RT],
    labels: LabelProvider[KT, LT],
    instance_viewer: Callable[
        [Instance[KT, DT, VT, RT]], Mapping[str, Any]
    ] = default_instance_viewer,
    label_viewer: Callable[
        [KT, LabelProvider[KT, LT]], Mapping[str, Any]
    ] = default_label_viewer,
    provider_hooks: Sequence[
        Callable[
            [InstanceProvider[IT, KT, DT, VT, RT]],
            Mapping[KT, Mapping[str, Any]],
        ]
    ] = list(),
) -> pd.DataFrame:
    def row_yielder() -> Iterator[Tuple[KT, Mapping[str, Any]]]:
        provider_result = [hook(provider) for hook in provider_hooks]
        for idx, ins in provider.items():
            key_prov_results = flatten_dicts(*[result[idx] for result in provider_result if idx in result])
            data_values = {**instance_viewer(ins), **label_viewer(idx, labels), **key_prov_results}
            yield idx, data_values
    results = dict(row_yielder())
    df = pd.DataFrame.from_dict(results, orient="index")
    return df
