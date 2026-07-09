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

from ..typehints.typevars import KT, VT
from typing import Generic, Mapping, Optional, Union
from uuid import UUID, uuid4

from typing import Any

from .memory import AbstractMemoryProvider, DataPoint


class StructuredTextInstance(DataPoint[Union[KT, UUID], Mapping[str, str], VT, str], Generic[KT, VT]):
    """An instance whose data is a mapping of named text data fields, e.g. the
    columns of a semi-structured record such as a form or a database row.

    Unlike :class:`~instancelib.instances.text.TextInstance`, which stores a
    single string as its data, a `StructuredTextInstance` stores a
    :class:`~typing.Mapping` from field name to field value (`data`). Its
    `representation` is automatically derived from this mapping by joining
    each ``key: value`` pair on a blank line, which makes it suitable for
    presentation to human annotators or as input to text-based models.

    Parameters
    ----------
    identifier : Union[KT, UUID]
        The identifier of the instance.
    data : dict[str, Any]
        A mapping of field names to their (textual) values.
    vector : Optional[VT]
        The optional vector representation of the instance.
    """

    def __init__(self, identifier: Union[KT, UUID], data: dict[str, Any], vector: Optional[VT] = None) -> None:
        super().__init__(
            identifier=identifier,
            data=data,
            vector=vector,
        )

    @property
    def representation(self) -> str:
        """Return a human-readable representation of the instance.

        Returns
        -------
        str
            The data of this instance formatted as ``key: value`` pairs,
            one per paragraph, in the order in which they appear in `data`.
        """
        return "\n\n".join([
            f"{key}: {value}" for key, value in self._data.items()
        ])



class StructuredTextInstanceProvider(
    AbstractMemoryProvider[StructuredTextInstance, Union[KT, UUID], Mapping[str, str], VT, str], Generic[KT, VT]
):
    """An in-memory :class:`~instancelib.instances.base.InstanceProvider` for
    :class:`StructuredTextInstance` objects.

    New instances created with :meth:`create` are automatically assigned a
    random :class:`~uuid.UUID` as identifier and added to the provider.
    """

    @staticmethod
    def construct(*args: Any, **kwargs: Any) -> StructuredTextInstance[KT, VT]:
        """Construct a new :class:`StructuredTextInstance` without adding it
        to the provider.

        Returns
        -------
        StructuredTextInstance[KT, VT]
            The newly constructed instance.
        """
        return StructuredTextInstance[KT, VT](*args, **kwargs)

    def create(self, *args: Any, **kwargs: Any) -> StructuredTextInstance[KT, VT]:  # type: ignore[override]
        """Construct a new :class:`StructuredTextInstance`, add it to this
        provider, and return it.

        Parameters
        ----------
        *args
            Positional arguments, forwarded to :class:`StructuredTextInstance`
            (typically `data` and, optionally, `vector`).
        **kwargs
            Keyword arguments, forwarded to :class:`StructuredTextInstance`.

        Returns
        -------
        StructuredTextInstance[KT, VT]
            The newly created and added instance, with a randomly generated
            :class:`~uuid.UUID` as its identifier.
        """
        new_key = uuid4()
        instance = StructuredTextInstance[KT, VT](new_key, *args, **kwargs)
        self.add(instance)
        return instance
