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

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, FrozenSet, Generic, Iterable, Sequence, TypeVar, Any, Tuple, Union

import numpy as np  # type: ignore
from scipy import stats  # type: ignore

from ..machinelearning.base import AbstractClassifier
from ..instances.base import Instance, InstanceProvider
from ..labels.base import LabelProvider
from ..labels.memory import MemoryLabelProvider

from ..utils.func import union

from ..typehints import KT, DT, VT, RT, LT

_T = TypeVar("_T")
class ResultUnit(Enum):
    PERCENTAGE = "Percentage"
    ABSOLUTE = "Absolute"
    FRACTION = "Fraction"

IT = TypeVar("IT", bound="Instance[Any,Any,Any,Any]", covariant = True)
InstanceInput = Union[InstanceProvider[IT, KT, DT, VT, RT], Iterable[Instance[KT, DT, VT, RT]]]


def instance_union(prov_func: Callable[[InstanceProvider[IT, KT, DT, VT, RT]], _T], 
                   iter_func: Callable[[Iterable[Instance[KT, DT, VT, RT]]], _T]
                   ) -> Callable[[InstanceInput[IT, KT, DT, VT, RT]], _T]:
    def wrapper(instances: InstanceInput[IT, KT, DT, VT, RT]) -> _T:
        if isinstance(instances, InstanceProvider):
            typed_input: InstanceProvider[IT, KT, DT, VT, RT] = instances
            return prov_func(typed_input)
        return iter_func(instances)
    return wrapper
      
def get_keys(instances: InstanceInput[IT, KT, DT, VT, RT]):
    def get_prov_keys(prov: InstanceProvider[IT, KT, DT, VT, RT]):
        return prov.key_list
    def get_all_keys(inss: Iterable[Instance[KT, DT, VT, RT]]):
        return [ins.identifier for ins in inss]
    return instance_union(get_prov_keys, get_all_keys)(instances)
                         



@dataclass
class BinaryModelMetrics(Generic[KT]):
    true_positives: FrozenSet[KT]
    true_negatives: FrozenSet[KT]
    false_positives: FrozenSet[KT]
    false_negatives: FrozenSet[KT]

    @property
    def recall(self) -> float:
        tp = len(self.true_positives)
        fn = len(self.false_negatives)
        recall = tp / ( tp + fn)
        return recall

    @property
    def precision(self) -> float:
        tp = len(self.true_positives)
        fp = len(self.false_positives)
        try:
            precision = tp / (tp + fp)
        except ZeroDivisionError:
            return 0.0
        return precision
    
    @property
    def accuracy(self) -> float:
        tp = len(self.true_positives)
        fp = len(self.false_positives)
        fn = len(self.false_negatives)
        tn = len(self.true_negatives)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        return accuracy

    @property
    def wss(self) -> float:
        tp = len(self.true_positives)
        fp = len(self.false_positives)
        fn = len(self.false_negatives)
        tn = len(self.true_negatives)
        n = tp + fp + fn + tn
        wss = ((tn + fn) / n) - (1 - (tp / ( tp + fn)))
        return wss

    @property
    def f1(self) -> float:
        return self.f_beta(beta=1)

    def f_beta(self, beta: int=1) -> float:
        b2 = beta*beta
        try:
            fbeta = (1 + b2) * (
                (self.precision * self.recall) /
                ((b2 * self.precision) + self.recall))
        except ZeroDivisionError:
            fbeta = 0.0
        return fbeta

class MulticlassModelMetrics(Generic[KT, LT]):
    def __init__(self, *label_performances: Tuple[LT, BinaryModelMetrics[KT]]):
        self.label_dict = {
            label: performance for (label, performance) in label_performances}
    
    @property
    def true_positives(self) -> FrozenSet[KT]:
        keys = union(*(pf.true_positives for pf in self.label_dict.values()))
        return keys

    @property
    def true_negatives(self) -> FrozenSet[KT]:
        keys = union(*(pf.true_negatives for pf in self.label_dict.values()))
        return keys

    @property
    def false_negatives(self) -> FrozenSet[KT]:
        keys = union(*(pf.false_negatives for pf in self.label_dict.values()))
        return keys
    
    @property
    def false_positives(self) -> FrozenSet[KT]:
        keys = union(*(pf.false_positives for pf in self.label_dict.values()))
        return keys

    @property
    def recall(self) -> float:
        tp = len(self.true_positives)
        fn = len(self.false_negatives)
        recall = tp / ( tp + fn)
        return recall

    @property
    def precision(self) -> float:
        tp = len(self.true_positives)
        fp = len(self.false_positives)
        precision = tp / (tp + fp)
        return precision
    
    @property
    def accuracy(self) -> float:
        tp = len(self.true_positives)
        fp = len(self.false_positives)
        fn = len(self.false_negatives)
        tn = len(self.true_negatives)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        return accuracy

    @property
    def f1(self) -> float:
        return self.f_beta(beta=1)

    def f_beta(self, beta: int=1) -> float:
        b2 = beta ** 2
        fbeta = (1 + b2) * (
            (self.precision * self.recall) /
            ((b2 * self.precision) + self.recall)) 
        return fbeta

    @property
    def f1_macro(self) -> float:
        return self.f_macro(beta=1)

    def f_macro(self, beta: float=1) -> float:
        average_recall: float = np.mean([pf.recall for pf in self.label_dict.values()]) # type: ignore
        average_precision: float = np.mean([pf.precision for pf in self.label_dict.values()]) # type: ignore
        b2 = beta ** 2
        fbeta = (1 + b2) * ((average_precision * average_recall) /
                ((b2 * average_precision) + average_recall)) 
        return fbeta
    
def label_metrics(truth: LabelProvider[KT, LT], 
                  prediction: LabelProvider[KT, LT], 
                  keys: Sequence[KT], label: LT):
    included_keys = frozenset(keys)
    ground_truth_pos = truth.get_instances_by_label(label).intersection(included_keys)
    pred_pos = prediction.get_instances_by_label(label)
    true_pos = pred_pos.intersection(ground_truth_pos)
    false_pos = pred_pos.difference(true_pos)
    false_neg = ground_truth_pos.difference(true_pos)
    true_neg = included_keys.difference(true_pos, false_pos, false_neg)
    return BinaryModelMetrics[KT](true_pos, true_neg, false_pos, false_neg)

def classifier_performance(model: AbstractClassifier[IT, KT, DT, VT, RT, LT, Any, Any], 
                           instances: InstanceInput[IT, KT, DT, VT, RT],
                           ground_truth: LabelProvider[KT, LT]
                           ) -> Dict[LT, BinaryModelMetrics[KT]]:
    keys = get_keys(instances)
    labelset = ground_truth.labelset
    predictions = model.predict(instances)
    pred_provider = MemoryLabelProvider.from_tuples(predictions)
    performance = {
        label: label_metrics(
            ground_truth, 
            pred_provider, 
            keys,
            label
        ) for label in labelset
    }
    return performance

def classifier_performance_mc(model: AbstractClassifier[IT, KT, DT, VT, RT, LT, Any, Any], 
                              instances: InstanceInput[IT, KT, DT, VT, RT],  
                              ground_truth: LabelProvider[KT, LT],) -> MulticlassModelMetrics[KT, LT]:
    keys = get_keys(instances)
    labelset = ground_truth.labelset
    predictions = model.predict(instances)
    pred_provider = MemoryLabelProvider.from_tuples(predictions)
    performances = [
        (label, label_metrics(ground_truth, 
                              pred_provider, 
                              keys, 
                              label)) for label in labelset]
    performance = MulticlassModelMetrics[KT, LT](*performances)    
    return performance