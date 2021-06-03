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

from typing import Generic, Optional, Sequence, Iterable, Union

from ..instances.text import TextInstanceProvider, TextInstance
from ..labels.memory import MemoryLabelProvider
from .memory import MemoryEnvironment

from uuid import UUID

from ..typehints import KT, VT, LT

class TextEnvironment(MemoryEnvironment[TextInstance[KT, VT], Union[KT, UUID], str, VT, str, LT], Generic[KT, VT, LT]):
    @classmethod
    def from_data(cls, 
                  target_labels: Iterable[LT], 
                  indices: Sequence[KT], 
                  data: Sequence[str], 
                  ground_truth: Sequence[Iterable[LT]],
                  vectors: Optional[Sequence[VT]]) -> TextEnvironment[KT, VT, LT]:
        dataset = TextInstanceProvider[KT, VT].from_data_and_indices(indices, data, vectors)
        truth = MemoryLabelProvider[Union[KT, UUID], LT].from_data(target_labels, indices, ground_truth)
        return cls(dataset, truth)
    



        


    
    
    



        

