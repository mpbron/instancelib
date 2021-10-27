from .environment.base import AbstractEnvironment, Environment
from .environment.memory import MemoryEnvironment
from .environment.text import TextEnvironment
from .ingest.spreadsheet import read_csv_dataset, read_excel_dataset, pandas_to_env, pandas_to_env_with_id
from .instances.base import Instance, InstanceProvider
from .instances.memory import DataPoint, DataPointProvider
from .instances.text import TextInstance, TextInstanceProvider
from .labels import LabelProvider
from .labels.memory import MemoryLabelProvider
from .machinelearning import SkLearnDataClassifier, SkLearnVectorClassifier

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
    "SkLearnDataClassifier", "SkLearnVectorClassifier",
    "read_csv_dataset", "read_excel_dataset", "pandas_to_env", "pandas_to_env_with_id"
]
