#%%
from instancelib.utils.func import list_unzip
import itertools
from instancelib.typehints.typevars import VT
from typing import Any, Callable, Iterable, Sequence

from instancelib import TextBucketProvider
from instancelib.ingest.spreadsheet import read_csv_dataset, read_excel_dataset
from instancelib.instances.text import TextInstance
from uuid import UUID, uuid4


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

class DeTokenizerWrapper:
    def __init__(self, detokenizer: Callable[[str], Sequence[str]]):
        self.detokenizer = detokenizer

class TokenPertubator:
    def __init__(self, 
                 detokenizer: Callable[[Iterable[str]], str], 
                 pertubator: Callable[[str], str]):
        self.detokenizer = detokenizer
        self.pertubator = pertubator

    def __call__(self, instance: TextInstance[Any, VT]) -> TextInstance[UUID, VT]:
        assert instance.tokenized
        new_tokenized = list(map(self.pertubator, instance.tokenized))
        new_data = self.detokenizer(new_tokenized)
        new_id = uuid4() # TODO This has to be improved. 
        # Current Provider architecture does not have functionality for **unique** new 
        # id generation. Or KT should be equal to UUID
        # (For Active Learning we did not need to create new Instances)
        new_instance = TextInstance[UUID, VT](new_id, new_data, None, new_data, new_tokenized)
        return new_instance
        


def tokenizer(input: str) -> Sequence[str]:
    return input.split(" ")

def detokenizer(input: Iterable[str]) -> str:
    return " ".join(input)

def dutch_article_pertubator(word: str) -> str:
    if word in ["de", "het"]:
        return "een"
    return word

wrapped_tokenizer = TokenizerWrapper(tokenizer)
pertubator = TokenPertubator(detokenizer, dutch_article_pertubator)
# %%

# Map this function over all instances in this provider 
# Mutates in place
instanceprovider.map_mutate(wrapped_tokenizer) # type: ignore
#%%
pertubated_instances = tweakers_env.create_empty_provider()
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
