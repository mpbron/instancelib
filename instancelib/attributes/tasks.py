from enum import Enum       

class MLTask(Enum):
    BINARY_CLASSIFICATION = "BinaryClassfication"
    MULTICLASS_CLASSIFICATION = "MulticlassClassification"
    MULTILABEL_CLASSIFICATION = "MultilabelClassification"
    REGRESSION = "Regression"
    