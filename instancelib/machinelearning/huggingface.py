from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Generic,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

import numpy as np
import numpy.typing as npt
from transformers import (
    Pipeline,
    PreTrainedModel,
    TextClassificationPipeline,
    TFPreTrainedModel,
    pipeline,
)
from typing_extensions import Self

from ..instances import Instance
from ..typehints.typevars import DT, KT, LMT, LT, LVT, PMT, RT, VT
from ..utils.func import invert_mapping
from .wrapper import (
    DataWrapper,
    numpy_mc_threshold,
    numpy_ova_threshold,
    to_int,
)

IT = TypeVar("IT", bound="Instance[Any, Any, Any, Any]")


@dataclass
class HuggingfaceWrapper:

    pipeline: Pipeline
    batch_size: int

    def __call__(self, x_data: Sequence[Any]) -> npt.NDArray[np.float64]:
        label_dict: Mapping[str, int] = invert_mapping(self.pipeline.model.config.id2label)  # type: ignore
        preds = self.pipeline(x_data, top_k=None, batch_size=self.batch_size)
        res_mat = np.zeros((len(preds), len(label_dict)))
        for i, pred in enumerate(preds):
            for lpred in pred:
                j = label_dict[lpred["label"]]
                s = lpred["score"]
                res_mat[i, j] = s
        return res_mat


class HuggingfaceClassifier(
    DataWrapper[IT, KT, DT, VT, RT, LT, LVT, LMT, PMT],
    Generic[IT, KT, DT, VT, RT, LT, LVT, LMT, PMT],
):
    _name = "Huggingface"

    @classmethod
    def build(
        cls,
        pipeline: Pipeline,
        threshold_f: Callable[[npt.NDArray[Any]], np.bool_],
        batch_size: int = 16,
        labels: Optional[Union[Sequence[LT], Mapping[int, LT]]] = None,
        **kwargs,
    ) -> Self:
        wrapped_pipeline = HuggingfaceWrapper(pipeline, batch_size)
        if labels is None:
            labels = dict(pipeline.model.config.id2label)  # type: ignore
        return cls.from_vectorized_function(
            wrapped_pipeline, to_int(threshold_f), labels  # type: ignore
        )  # type: ignore

    @classmethod
    def build_multilabel(
        cls,
        pipeline: Pipeline,
        batch_size: int = 16,
        threshold: float = 0.5,
        labels: Optional[Union[Sequence[LT], Mapping[int, LT]]] = None,
        **kwargs,
    ) -> Self:
        threshold_f = numpy_ova_threshold(threshold)
        model = cls.build(pipeline, to_int(threshold_f), batch_size, labels, **kwargs)  # type: ignore
        return model  # type: ignore

    @classmethod
    def build_multiclass(
        cls,
        pipeline: TextClassificationPipeline,
        batch_size: int = 16,
        labels: Optional[Union[Sequence[LT], Mapping[int, LT]]] = None,
        **kwargs,
    ) -> Self:
        threshold_f = numpy_mc_threshold
        model = cls.build(pipeline, to_int(threshold_f), batch_size, labels, **kwargs)  # type: ignore
        return model  # type: ignore

    def __repr__(self) -> str:
        result = (
            "HuggingfaceClassifier(" f"classes={list(self.encoder.labels)})"
        )
        return result

    @classmethod
    def from_pretrained(
        cls,
        model: Union[str, PreTrainedModel, TFPreTrainedModel],
        model_args: Mapping[str, Any] = dict(),
        batch_size: int = 16,
        **kwargs,
    ):
        return cls.build_multilabel(
            pipeline(model=model, **model_args), batch_size, **kwargs
        )
