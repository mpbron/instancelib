# Changelog
All notable changes to `instancelib` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
## [0.3.9.1]
### Bugfix
- Bugfix for on the fly encoding of data

## [0.3.9.0]
### Added
- Added create_subset_by_labels method that allows you to take a create a InstanceProvider based on labels
- Added a version of the SklearnDataClassifier that allows for on the fly encoding of data

## [0.3.8.0]
### Added
- Added preliminary support reading datasets in TREC qrel format
- Added several utility functions for handling probability matrices that only contain a single column

## [0.3.7.0]
### Added
- Confusion matrices for binary and multiclass performance analysis
- Added to_dict() method for instances.

## [0.3.6.2]
### Changed
- Bugfix in sklearn_model prediction function for when an empty list is provided as input.

## [0.3.6.1]
### Changed
- Bugfix in build_model class method for Sklearn models

## [0.3.6.0]
### Changed
- We now support models that output the full label in string format instead of categorical integer encoding.
- You can now convert the integer encoded labels from a sklearn model to string values, if that is necessary to match with your environment.

## [0.3.5.1]
### Changed
- Fixed a bug in which fitting a classifier failed if some instances did not have labels (Binary/Multiclass classifcation only).

## [0.3.5.0]
### Added
- Documentation for machinelearning subpackage
- More functionality available from top level import


## [0.3.4.4]
### Changed
- Bugfix: vectorize module was imported instead of vectorize function

## [0.3.4.3]
### Added
- Make Feature Extraction / Vectorization accessible from toplevel import

## [0.3.4.2]
### Changed
- Changed return type Environment to AbstractEnvironment in the pandas_to_env_with_id function
## [0.3.4.1] - 2021-10-27

### Added
- Updated documentation
- pandas_to_env_with_id function

[Unreleased]: https://github.com/mpbron/instancelib
[0.3.6.0]: https://pypi.org/project/instancelib/0.3.6.0
[0.3.5.1]: https://pypi.org/project/instancelib/0.3.5.0
[0.3.5.0]: https://pypi.org/project/instancelib/0.3.5.0
[0.3.4.4]: https://pypi.org/project/instancelib/0.3.4.4
[0.3.4.3]: https://pypi.org/project/instancelib/0.3.4.3
[0.3.4.2]: https://pypi.org/project/instancelib/0.3.4.2
[0.3.4.1]: https://pypi.org/project/instancelib/0.3.4.1
