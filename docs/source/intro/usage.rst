Basic usage
===========
On this page, we briefly introduce some of the main concepts of the architecture of the library.
The use case on this page is document classification, but InstanceLib is also suitable for other Machine Learning tasks and
not limited to text data.

Data Structure
--------------

You can easily read tabular data such as Excel files.
In the following example, we load a sample dataset of Dutch Tech news articles.

.. code:: python

    import instancelib as il

    text_env = il.read_excel_dataset("./datasets/testdataset.xlsx",
                                     data_cols=["fulltext"],
                                     label_cols=["label"])

We can access the data through an :class:`~instancelib.Environment` object.
An Environment contains two main components: :class:`~instancelib.InstanceProvider` and :class:`~instancelib.LabelProvider`. 
InstanceProviders contain :class:`~instancelib.Instance` objects (one for each raw data point). 
Each instance contains at least four properties:

1. ``identifier``. Each instance has a unique identifier that can be used to 
   retrieve the instance from the InstanceProvider.
2. ``data``. This contains the raw data. For this example problem, this
   property contains the text of the article.
3. ``vector``. Many classifiers cannot process the raw data directly if they
   are not numerical. In this property, we can store a numerical vector that
   represents the raw data. In this example, we can store a TF-IDF vector
   representation or any otherdocument embedding. This property can be ``None``.
4. ``representation``. In some cases, the human-readable format differs from
   the raw data. If this is the case, the human-readable representation can be
   retrieved by this property.


InstanceProviders are `dictionary-like` objects that store these instances. 
You can easily retrieve each document by its identifier. 
:class:`~instancelib.Environment` objects may contain several InstanceProvider objects.
The most important one is the ``dataset`` which contains all datapoints.
This one can be retrieved as follows:

>>> ins_provider = text_env.dataset

Then, you can use standard :class:`dict` methods to access the individual Instances.

>>> ins  = ins_provider[20]
>>> ins.data
"Super Smash Brothers voor de Wii U ..."

For this example dataset, we already have labeled data points (they are stored in the `label` column in the Excel file).
The :class:`~instancelib.LabelProvider` records the labels or classes of each instance. 
You can request the labels for this instance as follows:

>>> text_env.labels.get_labels(ins)
frozenset({"Games"})

The labels are returned as :class:`frozenset` objects to be able to deal with Multilabel Classification problems.

You can also request the keys for all documents that have the label `Games`.

>>> text_env.labels.get_instances_by_label("Games")
frozenset({0, 1, 2, 3, ... , 97})

Often, you do want to divide your dataset further into subsets.
In classification, we test the performance of a model by using a held out test set. 
We can create a random `train test split` of 70 % train and 30 % test as follows.

>>> train, test = text_env.train_test_split(ins_provider)

Machine Learning
----------------

End-to-end models
^^^^^^^^^^^^^^^^^

You can also train models with instancelib.
These models can be based on either the data parts or the vectors of the instances.
Scikit-learn offers end-to-end models in the form of :class:`~sklearn.pipeline.Pipeline` objects.
The :class:`~instancelib.SkLearnDataClassifier` object allows you to use the end-to-end models tbat 
adhere to the Scikit-learn API. 
The model uses the :meth:`~instancelib.Instance.data` property directly and ignores the vectors.

.. code:: python

    from sklearn.pipeline import Pipeline 
    from sklearn.naive_bayes import MultinomialNB 
    from sklearn.feature_extraction.text import TfidfTransformer

    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
        ])

    model = il.SkLearnDataClassifier.build(pipeline, text_env)
    model.fit_provider(train, labels)
    predictions = model.predict(test)

The :func:`~instancelib.machinelearning.sklearn.SkLearnClassifier.build` method handles translation between the Scikit-learn API and
and the :class:`~instancelib.Environment` objects for binary and multiclass problems. 
If your problem is multilabel classification, then you can use the :func:`~instancelib.machinelearning.sklearn.SkLearnClassifier.build_multilabel` method.


Vector-based models
^^^^^^^^^^^^^^^^^^^

Instead of using the raw data directly, you can use your own feature extraction or embeddings.
For document classification, it would make sense to use a document embedding like Doc2Vec.
In this library, you can easily integrate them in your pipeline.
First, we define the feature extraction method:

.. code:: python
   
   from instancelib.feature_extraction.doc2vec import Doc2VecVectorizer
   d2v = il.TextInstanceVectorizer(Doc2VecVectorizer())

Vectorizer objects first need to be fitted on some data to produce vectors. 

Then, you can give all documents the corresponding vectors by issuing the following line:

>>> vectorize(d2v, text_env)

Now every instance in the ``text_env`` has a vector. Next, we can define a new model that can be used to 
classify the documents. 

Similar as in the end-to-end example, we can construct a new model by using the ``build`` method.

.. code:: python

   from sklearn.svm import SVC
   svm = SVC(kernel="linear", probability=True, class_weight="balanced")
   vec_model = il.SkLearnVectorClassifier.build(svm, text_env)

   vec_model.fit_provider(train, text_env.labels)
   predictions = vec_model.predict(test)


