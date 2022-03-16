import instancelib as il
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

DATASET_FILE = "datasets/testdataset.xlsx"

def test_dataloading():
    env = il.read_excel_dataset(DATASET_FILE, ["fulltext"], ["label"])
    ins20 = env.dataset[20]
    train, test = env.train_test_split(env.dataset, 0.70)
    assert ins20.identifier == 20
    assert env.labels.get_labels(ins20) == frozenset({"Games"})
    assert all((ins not in test for ins in train ))

def test_vectorizing():
    env = il.read_excel_dataset(DATASET_FILE, ["fulltext"], ["label"])
    vect = il.TextInstanceVectorizer(il.SklearnVectorizer(TfidfVectorizer(max_features=1000)))
    il.vectorize(vect, env)
    assert env.dataset[20].vector is not None
    assert env.dataset[20].vector.shape == (1000,)

def test_classification():
    env = il.read_excel_dataset(DATASET_FILE, ["fulltext"], ["label"])
    vect = il.TextInstanceVectorizer(il.SklearnVectorizer(TfidfVectorizer(max_features=1000)))
    il.vectorize(vect, env)
    train, test = env.train_test_split(env.dataset, 0.70)
    model = il.SkLearnVectorClassifier.build(MultinomialNB(), env)
    model.fit_provider(train, env.labels)
    performance = il.classifier_performance(model, test, env.labels)
    assert performance["Games"].f1 >= 0.75
    assert performance["Smartphones"].f1 >= 0.75
    assert performance["Bedrijfsnieuws"].f1 >= 0.75

def test_build_from_model():
    env = il.read_excel_dataset(DATASET_FILE, ["fulltext"], ["label"])
    vect = il.TextInstanceVectorizer(il.SklearnVectorizer(TfidfVectorizer(max_features=1000)))
    il.vectorize(vect, env)
    train, test = env.train_test_split(env.dataset, 0.70)
    model = il.SkLearnVectorClassifier.build(MultinomialNB(), env)
    model.fit_provider(train, env.labels)
    build_model = il.SkLearnVectorClassifier.build_from_model(model.innermodel, ints_as_str=True) # type: ignore
    model_preds = model.predict(test)
    b_model_preds = build_model.predict(test) # type: ignore
    for (a, pred_lt), (b, pred_idx) in zip(model_preds, b_model_preds):
        assert a == b
        first_label_lt = next(iter(pred_lt))
        first_label_idx = next(iter(pred_idx))
        assert str(model.get_label_column_index(first_label_lt)) == first_label_idx

def test_confmat():
    env = il.read_excel_dataset(DATASET_FILE, ["fulltext"], ["label"])
    vect = il.TextInstanceVectorizer(il.SklearnVectorizer(TfidfVectorizer(max_features=1000)))
    il.vectorize(vect, env)
    train, test = env.train_test_split(env.dataset, 0.70)
    model = il.SkLearnVectorClassifier.build(MultinomialNB(), env)
    model.fit_provider(train, env.labels)
    model_preds = model.predict(test)
    preds = il.MemoryLabelProvider.from_tuples(model_preds)
    conf_mat = il.analysis.confusion_matrix(env.labels, preds, test)
    print(conf_mat)

def test_pandas_multiple():
    df = pd.read_excel(DATASET_FILE)
    env = il.pandas_to_env({"train": df, "test": df}, ["fulltext"], ["label"])
    vect = il.TextInstanceVectorizer(il.SklearnVectorizer(TfidfVectorizer(max_features=1000)))
    il.vectorize(vect, env)
    assert env["train"]
    assert env["test"]
    assert env["train"]["train_20"].data == env["test"]["test_20"].data
    assert env.labels.get_labels(env["train"]["train_20"])
    assert env.labels.get_labels(env["train"]["train_20"]) == env.labels.get_labels(env["test"]["test_20"])
    assert env.labels.get_labels(env["train"]["train_20"]) == frozenset({"Games"})
    assert len(env["train"]) == len(env["test"])
    train, test = env.train_test_split(env.dataset, 0.70)
    model = il.SkLearnVectorClassifier.build(MultinomialNB(), env)
    model.fit_provider(train, env.labels)
    model_preds = model.predict(test)
    preds = il.MemoryLabelProvider.from_tuples(model_preds)
    conf_mat = il.analysis.confusion_matrix(env.labels, preds, test)
    print(conf_mat)

def test_pandas_single():
    df = pd.read_excel(DATASET_FILE)
    env = il.pandas_to_env(df, ["fulltext"], ["label"])
    vect = il.TextInstanceVectorizer(il.SklearnVectorizer(TfidfVectorizer(max_features=1000)))
    il.vectorize(vect, env)
    train, test = env.train_test_split(env.dataset, 0.70)
    model = il.SkLearnVectorClassifier.build(MultinomialNB(), env)
    model.fit_provider(train, env.labels)
    model_preds = model.predict(test)
    preds = il.MemoryLabelProvider.from_tuples(model_preds)
    conf_mat = il.analysis.confusion_matrix(env.labels, preds, test)
    print(conf_mat)

def test_usage():
    from instancelib import TextEnvironment
    import instancelib as il
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
    from instancelib.ingest.spreadsheet import read_excel_dataset
    from instancelib.instances.text import TextInstance
    from instancelib.pertubations.base import TokenPertubator

    def binary_mapper(value: Any) -> str:
        if value == 1:
            return "Relevant"
        return "Irrelevant"
        
    text_env = read_excel_dataset(DATASET_FILE,
                                    data_cols=["fulltext"],
                                    label_cols=["label"])

    text_env
    #%%
    ins_provider = text_env.dataset
    labelprovider = text_env.labels

    #%%
    n_docs = len(ins_provider)
    n_train = round(0.70 * n_docs)
    train, test = text_env.train_test_split(ins_provider, train_size=0.70)

    #%% 
    text_env["train"], text_env["test"] = train, test
    #%%
    # Test if we indeed got the right length
    print((len(train) == n_train))
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
    pertubated_instances = text_env.create_empty_provider()

    #%%
    wrapped_tokenizer = TokenizerWrapper(tokenizer)
    pertubator = TokenPertubator(
        text_env, tokenizer, detokenizer, dutch_article_pertubator)
    #%%
    ins_provider.map_mutate(wrapped_tokenizer) 

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

    # %%
    vectorizer = TextInstanceVectorizer(
        il.SklearnVectorizer(TfidfVectorizer(max_features=1000)))

    vectorize(vectorizer, text_env)
    #%%
    classifier = MultinomialNB()
    vec_model = SkLearnVectorClassifier.build(classifier, text_env)

    #%%
    vec_model.fit_provider(train, text_env.labels)
    # %%

    docs = list(test.instance_chunker(20))[0]

    #%%

    predictions = vec_model.predict(docs)
    # %%
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
        ])
    data_model = SkLearnDataClassifier.build(pipeline, text_env)
    # %%tweakers_env#%%
    env = TextEnvironment.from_data(["A", "B", "C"], [1,2,3], ["Test", "Test2", "Test3"], [["A"], ["A", "B"], ["C"]], None)


    