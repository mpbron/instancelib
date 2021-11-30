import instancelib as il
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
    build_model = il.SkLearnVectorClassifier.build_from_model(model.innermodel, ints_as_str=True)
    model_preds = model.predict(test)
    b_model_preds = build_model.predict(test)
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

    