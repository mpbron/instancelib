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

from typing import Sequence, Any

import numpy as np # type: ignore

from ..instances import Instance

from .base import BaseVectorizer

InstanceList = Sequence[Instance[Any, str, np.ndarray, Any]] # type: ignore

class TextInstanceVectorizer(BaseVectorizer[Instance[Any, str, np.ndarray, Any]]): 
    _name = "TextInstanceVectorizer"
    def __init__(self,
                 vectorizer: BaseVectorizer[str],
                 ) -> None:
        super().__init__()
        self.innermodel = vectorizer

    @property
    def fitted(self) -> bool:
        return self.innermodel.fitted

    def fit(self, x_data: InstanceList, **kwargs: Any) -> TextInstanceVectorizer:
        texts = [x.data for x in x_data]
        self.innermodel.fit(texts)
        return self

    def transform(self, x_data: InstanceList, **kwargs: Any) -> np.ndarray:
        texts = [x.data for x in x_data]
        return self.innermodel.transform(texts) # type: ignore

    def fit_transform(self, x_data: InstanceList, **kwargs: Any) -> np.ndarray:
        texts = [x.data for x in x_data]
        return self.innermodel.fit_transform(texts) # type: ignore
