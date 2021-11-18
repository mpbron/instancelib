**<h3 align="center">
A generic dataset interface for Machine Learning models**
</h3>

[![PyPI](https://img.shields.io/pypi/v/instancelib)](https://pypi.org/project/instancelib/)
[![Python_version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)](https://pypi.org/project/instancelib/)
[![License](https://img.shields.io/pypi/l/instancelib)](https://www.gnu.org/licenses/lgpl-3.0.en.html)

---

`instancelib` provides a **generic architecture** for datasets. 

&copy; Michiel Bron, 2021

## Quick tour
**Load dataset**: Load the dataset in an environment
```python
import instancelib as il
text_env = il.read_excel_dataset("./datasets/testdataset.xlsx",
                                  data_cols=["fulltext"],
                                  label_cols=["label"])

ds = text_env.dataset # A `dict-like` interface for instances
labels = text_env.labels # An object that stores all labels
labelset = labels.labelset # All labels that can be given to instances

ins = ds[20] # Get instance with identifier key  `20`
ins_data = ins.data # Get the raw data for instance 20
ins_vector = ins.vector # Get the vector representation for 20 if any

ins_labels = labels.get_labels(ins)
``` 

**Dataset manipulation**: Divide the dataset in a train and test set
```python
train, test = text_env.train_test_split(ds, train_size=0.70)

print(20 in train) # May be true or false, because of random sampling
```

**Train a model**:
```python
from sklearn.pipeline import Pipeline 
from sklearn.naive_bayes import MultinomialNB 
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

pipeline = Pipeline([
     ('vect', CountVectorizer()),
     ('tfidf', TfidfTransformer()),
     ('clf', MultinomialNB()),
     ])

model = il.SkLearnDataClassifier.build(pipeline, text_env)
model.fit_provider(train, labels)
predictions = model.predict(test)
```
## Installation
See [installation.md](docs/installation.md) for an extended installation guide.

| Method | Instructions |
|--------|--------------|
| `pip` | Install from [PyPI](https://pypi.org/project/instancelib/) via `pip install instancelib`. |
| Local | Clone this repository and install via `pip install -e .` or locally run `python setup.py install`.

## Documentation
Full documentation of the latest version is provided at [https://instancelib.readthedocs.org](https://instancelib.readthedocs.org).

## Example usage
See [usage.py](usage.py) to see an example of how the package can be used.

## Releases
`instancelib` is officially released through [PyPI](https://pypi.org/project/instancelib/).

See [CHANGELOG.md](CHANGELOG.md) for a full overview of the changes for each version.

## Citation
```bibtex
@misc{instancelib,
  title = {Python package instancelib},
  author = {Michiel Bron},
  howpublished = {\url{https://github.com/mpbron/instancelib}},
  year = {2021}
}
```
## Library usage
This library is used in the following projects:
- [python-allib](https://github.com/mpbron/allib). A typed Active Learning framework for Python for both Classification and Technology-Assisted Review systems.
- [text_explainability](https://marcelrobeer.github.io/text_explainability/). A generic explainability architecture for explaining text machine learning models
- [text_sensitivity](https://marcelrobeer.github.io/text_sensitivity/). Sensitivity testing (fairness & robustness) for text machine learning models.

## Maintenance
### Contributors
- [Michiel Bron](https://www.uu.nl/staff/MPBron) (`@mpbron`)

### Todo
Tasks yet to be done:

* Implement support for ONNX models
* Implement support for Python DataLoaders
* Make the external dataset interface more user friendly
* Redesign LabelProvider to support more attribute levels
* CI/CD tests