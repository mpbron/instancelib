#%%
from functools import reduce
from instancelib.typehints.typevars import KT, VT
from instancelib.machinelearning.skdata import SkLearnDataClassifier
from instancelib.machinelearning import SkLearnVectorClassifier
from typing import Any, Callable, Iterable, Sequence

from sklearn.pipeline import Pipeline # type: ignore
from sklearn.naive_bayes import MultinomialNB # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer # type: ignore

from instancelib.feature_extraction.textinstance import TextInstanceVectorizer
from instancelib.feature_extraction.textsklearn import SklearnVectorizer
from instancelib.functions.vectorize import vectorize
from instancelib.ingest.spreadsheet import read_csv_dataset, read_excel_dataset
from instancelib.instances.text import TextInstance
from instancelib.pertubations.base import TokenPertubator


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
                                   data_cols=["title", "abstract"], 
                                   label_cols=["included"], label_mapper=binary_mapper)

#%%
instanceprovider = tweakers_env.dataset
labelprovider = tweakers_env.labels

#%%
n_docs = len(instanceprovider)
n_train = round(0.70 * n_docs)
train, test = tweakers_env.train_test_split(instanceprovider, train_size=n_train)

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

    def __call__(self, instance: TextInstance[KT, VT]) -> TextInstance[KT, VT]:
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
pertubator = TokenPertubator(
    tweakers_env, tokenizer, detokenizer, dutch_article_pertubator)
#%%
instanceprovider.map_mutate(wrapped_tokenizer) 

#%%
#Pertubate an instance
assert isinstance(instance, TextInstance)
instance.tokenized

new_instance = pertubator(instance)
#%%
pertubated_instances.add(new_instance)
#%%
pertubated_instances.add_child(instance, new_instance)
#%%
pertubated_instances.get_parent(new_instance)
pertubated_instances.get_children(instance)

#%%
# Perform the pertubation on all test data
pertubated_test_data = frozenset(test.map(pertubator))

#%%
#%%
# Add the data to the test set
# add_range is type safe with * expansion from immutable data structures like frozenset, tuple, sequence
# But works with other data structures as well
test.add_range(*pertubated_test_data)

# %%
vectorizer = TextInstanceVectorizer(
    SklearnVectorizer(
        TfidfVectorizer(max_features=1000)))

vectorize(vectorizer, tweakers_env)
#%%
classifier = MultinomialNB()
vec_model = SkLearnVectorClassifier.build(classifier, tweakers_env)
#%%
vec_model.fit_provider(train, tweakers_env.labels)
# %%
predictions = vec_model.predict_proba(test)
# %%
pipeline = Pipeline([
     ('vect', CountVectorizer()),
     ('tfidf', TfidfTransformer()),
     ('clf', MultinomialNB()),
     ])
data_model = SkLearnDataClassifier.build(pipeline, tweakers_env)
# %%
data_model.fit_provider(train, tweakers_env.labels)
#%%
predictions_data = data_model.predict(test)

#%%