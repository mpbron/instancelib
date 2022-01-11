Datasets
========

To perform a s


Data Format
-----------

To use a dataset with InstanceLib on your own dataset, your data
file needs to adhere to a certain format. InstanceLib accepts the following
formats:

 - **Tabular Datasets** with extensions ``.csv``, ``.xlsx``,
   or ``.xls``. CSV and TAB files are preferably comma, semicolon, or tab-delimited.
   The preferred file encoding is *UTF-8*. These can be imported with the functions
   :func:`~instancelib.ingest.spreadsheet.read_csv_dataset` and 
   :func:`~instancelib.ingest.spreadsheet.read_excel_dataset` respectively.
 - **Pandas DataFrames**. If you already created a :class:`~pandas.DataFrame` with all your data,
    you can directly use the :func:`~instancelib.ingest.spreadsheet.build_environment` function to
    convert your DataFrame to an :class:`~instancelib.Environment` object that can be used further.

