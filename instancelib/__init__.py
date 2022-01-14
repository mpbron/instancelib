from .environment.base import AbstractEnvironment, Environment
from .environment.memory import MemoryEnvironment
from .environment.text import TextEnvironment
from .feature_extraction import (BaseVectorizer, SklearnVectorizer,
                                 TextInstanceVectorizer)
from .functions.vectorize import vectorize
from .ingest.spreadsheet import (pandas_to_env, pandas_to_env_with_id,
                                 read_csv_dataset, read_excel_dataset)
from .instances.base import Instance, InstanceProvider
from .instances.memory import DataPoint, DataPointProvider
from .instances.text import TextInstance, TextInstanceProvider
from .labels import LabelProvider
from .labels.memory import MemoryLabelProvider
from .machinelearning import SkLearnDataClassifier, SkLearnVectorClassifier, AbstractClassifier, SeparateDataEncoderClassifier
from .analysis import classifier_performance_mc, classifier_performance

__author__ = "Michiel Bron"
__email__ = "m.p.bron@uu.nl"

__all__= [
    "Instance", "InstanceProvider", 
    "DataPointProvider", "DataPoint",
    "TextInstance", "TextInstanceProvider",
    "AbstractEnvironment", "MemoryEnvironment",
    "Environment",
    "TextEnvironment",
    "LabelProvider",
    "MemoryLabelProvider",
    "SkLearnDataClassifier", "SkLearnVectorClassifier", "AbstractClassifier", "SeparateDataEncoderClassifier",
    "read_csv_dataset", "read_excel_dataset", "pandas_to_env", "pandas_to_env_with_id", 
    "vectorize", "BaseVectorizer", "SklearnVectorizer", "TextInstanceVectorizer",
    "classifier_performance", "classifier_performance_mc"
]
