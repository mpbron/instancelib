# Copyright (C) 2021 The InstanceLib Authors. All Rights Reserved.

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 3 of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

from __future__ import annotations

from typing import Sequence, Optional, Any

import numpy as np 

from sklearn.base import BaseEstimator # type: ignore
from sklearn.exceptions import NotFittedError # type: ignore

from ..utils import SaveableInnerModel
from .base import BaseVectorizer


class SklearnVectorizer(BaseVectorizer[str], SaveableInnerModel):
    innermodel: BaseEstimator
    _name = "SklearnVectorizer"

    def __init__(
        self,
        vectorizer: BaseEstimator,
        storage_location: Optional[str] = None,
        filename: Optional[str] = None
        ) -> None:
        BaseVectorizer.__init__(self) #type: ignore
        SaveableInnerModel.__init__(
            self, vectorizer, storage_location, filename)

    @SaveableInnerModel.load_model_fallback
    def fit(self, x_data: Sequence[str], **kwargs: Any) -> SklearnVectorizer:
        self.innermodel = self.innermodel.fit(x_data) # type: ignore
        self._fitted = True
        return self

    @SaveableInnerModel.load_model_fallback
    def transform(self, x_data: Sequence[str], **kwargs: Any) -> np.ndarray: # type: ignore
        if self.fitted:
            # TODO Check for performance issues with .toarray()
            return self.innermodel.transform(x_data).toarray() # type: ignore
        raise NotFittedError

    def fit_transform(self, x_data: Sequence[str], **kwargs: Any) -> np.ndarray: # type: ignore
        self.fit(x_data)
        return self.transform(x_data) # type: ignore
    