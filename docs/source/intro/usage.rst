Basic usage
===========

.. code:: python

    import instancelib as il

    tweakers_env = il.read_excel_dataset("dataset.xlsx",
                                  data_cols=["fulltext"],
                                  label_cols=["label"])
    halldataset_env = il.read_csv_dataset("./datasets/Software_Engineering_Hall.csv", 
                                   data_cols=["title", "abstract"], 
                                   label_cols=["included"], label_mapper=binary_mapper)
