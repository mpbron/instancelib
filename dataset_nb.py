#%%
from typing import Any
import instancelib as il
import pandas as pd
from instancelib.feature_extraction.base import DT
from instancelib.instances.dataset import PandasDataset, ReadOnlyProvider
from instancelib.typehints.typevars import KT
# %%
df = pd.read_csv("./datasets/Software_Engineering_Hall.csv")
# %%
ds = PandasDataset(df, "abstract")

# %%
def builder(identifier: KT, data: DT) -> il.DataPoint[KT, DT, Any, Any]:
    return il.DataPoint(identifier, data, None, data)
# %%
prov = ReadOnlyProvider(ds, builder)
# %%
