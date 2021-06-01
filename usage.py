#%%
from instancelib.pertubations.base import TokenPertubator
import itertools
from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, Iterable, Sequence, TypeVar, Union
from uuid import UUID

import numpy as np

from instancelib import TextBucketProvider
from instancelib.ingest.spreadsheet import read_csv_dataset, read_excel_dataset
from instancelib.instances import Instance
from instancelib.instances.text import TextInstance, TextInstanceProvider
from instancelib.typehints.typevars import KT, VT
from instancelib.utils.func import list_unzip





#%%
def binary_mapper(value: Any) -> str:
    if value == 1:
        return "Relevant"
    return "Irrelevant"
    
#%%
tweakers_env = read_excel_dataset("./datasets/testdataset.xlsx",
                                  data_cols=["fulltext"],
                                  label_cols=["label"])
halldataset_env = read_csv_dataset("./datasets/Software_Engineering_Hall.csv", 
                                   data_cols=["title", "abstract"], label_cols=["included"], label_mapper=binary_mapper)

#%%
instanceprovider = tweakers_env.dataset
labelprovider = tweakers_env.labels

#%%
n_docs = len(instanceprovider)
n_train = round(0.70 * n_docs)
train, test = TextBucketProvider.train_test_split(instanceprovider, train_size=n_train)

#%%
# Test if we indeed got the right length
len(train) == n_train
#%%
# Test if the train and test set are mutually exclusive
all([doc not in test for doc in train])
                                
#%% 
# Get the first document within training
key, instance = next(iter(train.items()))
print(instance)

# %% 
# Get the label for document
labelprovider.get_labels(instance)

# %%
# Get all documents with label "Bedrijfsnieuws"
bedrijfsnieuws_ins = labelprovider.get_instances_by_label("Bedrijfsnieuws")

# %%
# Get all training instances with label bedrijfsnieuws
bedrijfsnieuws_train = bedrijfsnieuws_ins.intersection(train)


# %%
# Some Toy examples
class TokenizerWrapper:
    def __init__(self, tokenizer: Callable[[str], Sequence[str]]):
        self.tokenizer = tokenizer

    def __call__(self, instance: TextInstance[Any, VT]) -> TextInstance[Any, VT]:
        data = instance.data
        tokenized = self.tokenizer(data)
        instance.tokenized = tokenized
        return instance


        
# %%
# Some function that we want to use on the instancess
def tokenizer(input: str) -> Sequence[str]:
    return input.split(" ")

def detokenizer(input: Iterable[str]) -> str:
    return " ".join(input)

def dutch_article_pertubator(word: str) -> str:
    if word in ["de", "het"]:
        return "een"
    return word

# %%
pertubated_instances = tweakers_env.create_empty_provider()

#%%
wrapped_tokenizer = TokenizerWrapper(tokenizer)
pertubator = TokenPertubator[int, np.ndarray](
    pertubated_instances, tokenizer, detokenizer, dutch_article_pertubator) # type: ignore
#%%
instanceprovider.map_mutate(wrapped_tokenizer) # type: ignore
#%%
#Pertubate an instance
assert isinstance(instance, TextInstance)
instance.tokenized

new_instance = pertubator(instance) # type: ignore
#%%
pertubated_instances.add(new_instance) # type: ignore
#%%
pertubated_instances.add_child(instance, new_instance) # type: ignore

#%%
pertubated_instances.get_parent(new_instance) # type: ignore
pertubated_instances.get_children(instance) # type: ignore



#%%
# Fitting a machine learning model (Assuming vectors are available)
batch_size = 50
key_vector_pairs = itertools.chain.from_iterable(train.vector_chunker(batch_size))
keys, vectors = list_unzip(key_vector_pairs)
labelings = list(map(labelprovider.get_labels, keys))
# classifier.fit_vectors(vectors, labelings) # Cf. allib


# %%
