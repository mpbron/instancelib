import instancelib as il
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def test_dataloading():
    env = il.read_excel_dataset("datasets/testdataset.xlsx", ["fulltext"], ["label"])
    ins20 = env.dataset[20]
    train, test = env.train_test_split(env.dataset, 0.70)
    assert ins20.identifier == 20
    assert env.labels.get_labels(ins20) == frozenset({"Games"})
    assert all((ins not in test for ins in train ))

def test_vectorizing():
    env = il.read_excel_dataset("datasets/testdataset.xlsx", ["fulltext"], ["label"])
    vect = il.TextInstanceVectorizer(il.SklearnVectorizer(TfidfVectorizer(max_features=1000)))
    il.vectorize(vect, env)
    assert env.dataset[20].vector is not None
    assert env.dataset[20].vector.shape == (1000,)

def test_classification():
    env = il.read_excel_dataset("datasets/testdataset.xlsx", ["fulltext"], ["label"])
    vect = il.TextInstanceVectorizer(il.SklearnVectorizer(TfidfVectorizer(max_features=1000)))
    il.vectorize(vect, env)
    train, test = env.train_test_split(env.dataset, 0.70)
    model = il.SkLearnVectorClassifier.build(MultinomialNB(), env)
    model.fit_provider(train, env.labels)
    performance = il.classifier_performance(model, test, env.labels)
    assert performance["Games"].f1 >= 0.75
    assert performance["Smartphones"].f1 >= 0.75
    assert performance["Bedrijfsnieuws"].f1 >= 0.75