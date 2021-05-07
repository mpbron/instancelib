# InstanceLib

An abstract interface for datasets.
This library is hosted on [https://git.science.uu.nl/mpbron-phd/instancelib](https://git.science.uu.nl/mpbron-phd/instancelib) and available for usage according to the GNU LGPL v3 license. 
This library is derived from the data management part of [Allib](https://git.science.uu.nl/mpbron-phd/allib). 

# Installation
Clone this repository and then execute the following command within the repo directory:
`pip install -e "./"`.

# Usage
See `usage.py` for examples on how to use this library. 

# TODO: 
- Tokenization mapping (which characters were removed)
- Function f: Instance -> LimePertubation -> Sequence[Instance']
- Machine Learning Wrappers that can work with InstanceProviders (cf. allib)
- Feature Extraction (cf. allib?)
- Method that converts One-hot Encoding of labels to Human Readable format


