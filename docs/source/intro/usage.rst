Basic usage
===========

You can easily read tabular data such as Excel files.
In the following example, we load a sample dataset of Dutch Tech news articles.
The code will be loaded in an :class:`Environment`.

.. code:: python

    import instancelib as il

    text_env = il.read_excel_dataset("./datasets/testdataset.xlsx",
                                     data_cols=["fulltext"],
                                     label_cols=["label"])

An Environment contains two main components: :class:`InstanceProvider` and :class:`LabelProvider`. 
InstanceProviders contain :class:`Instance` objects (one for each raw data point). 
Each instance contains at least four properties:

1. ``identifier``. Each instance has a unique identifier that can be used to retrieve the instance from the InstanceProvider.
2. ``data``. This contains the raw data. For this example problem, this property contains the text of the article.
3. ``vector``. Many classifiers cannot process the raw data directly if they are not numerical. In this property, we can store a numerical vector that represents the raw data. In this example, we can store a TF-IDF vector representation or document embedding.
4. ``representation``. In some cases, the human-readable format differs from the raw data. If this is the case, the human-readable representation can be retrieved by this property.


InstanceProviders are `dictionary-like` objects that store these instances. 
You can easily retrieve each document by its identifier. 

.. code:: python 

    ins_provider = text_env.dataset
    labelprovider = text_env.labels


>>> ins_provider[20].data
"Super Smash Brothers voor de Wii U ..."