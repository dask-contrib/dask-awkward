import math

import numpy as np

import awkward as ak
from awkward._nplikes import nplike_of
from awkward.forms import (
    Form,
    EmptyForm,
    NumpyForm,
    BitMaskedForm,
    ByteMaskedForm,
    IndexedForm,
    IndexedOptionForm,
    ListForm,
    ListOffsetForm,
    RegularForm,
    UnmaskedForm,
    RecordForm,
    UnionForm,
)
from awkward.contents import (
    Content,
    EmptyArray,
    NumpyArray,
    BitMaskedArray,
    ByteMaskedArray,
    IndexedArray,
    IndexedOptionArray,
    ListArray,
    ListOffsetArray,
    RegularArray,
    UnmaskedArray,
    RecordArray,
    UnionArray,
)


class DummyIndex:
    def __init__(self, length, nplike):
        self._length = length
        self._nplike = nplike

    def __len__(self):
        return self._length

    @property
    def length(self):
        return self._length


class DummyIndex8(DummyIndex, ak.index.Index8):
    @property
    def dtype(self):
        return np.dtype(np.int8)


class DummyIndexU8(DummyIndex, ak.index.Index8):
    @property
    def dtype(self):
        return np.dtype(np.uint8)


class DummyIndex32(DummyIndex, ak.index.Index8):
    @property
    def dtype(self):
        return np.dtype(np.int32)


class DummyIndexU32(DummyIndex, ak.index.Index8):
    @property
    def dtype(self):
        return np.dtype(np.uint32)


class DummyIndex64(DummyIndex, ak.index.Index8):
    @property
    def dtype(self):
        return np.dtype(np.int64)


index_of = {
    "i8": DummyIndex8,
    "u8": DummyIndexU8,
    "i32": DummyIndex32,
    "u32": DummyIndexU32,
    "i64": DummyIndex64,
}
dtype_of = {
    "i32": np.int32,
    "u32": np.uint32,
    "i64": np.int64,
}


def dummy_buffer(shape, dtype, nplike):
    return nplike.broadcast_to(nplike.asarray([0], dtype=dtype), shape)


def compatible(form: Form, layout: Content) -> bool:
    if layout is None:
        return True

    elif isinstance(layout, Content) and type(form) is layout.form_cls:
        if isinstance(form, (EmptyForm, NumpyForm)):
            # 0 contents
            return True

        elif isinstance(
            form,
            (
                BitMaskedForm,
                ByteMaskedForm,
                IndexedForm,
                IndexedOptionForm,
                ListForm,
                ListOffsetForm,
                RegularForm,
                UnmaskedForm,
            ),
        ):
            # 1 content
            return compatible(form.content, layout.content)

        elif isinstance(form, RecordForm):
            # arbitrarily many contents, possibly with missing fields
            for field in form.fields:
                if layout.has_field(field):
                    if not compatible(form.content(field), layout.content(field)):
                        return False
            return True

        elif isinstance(form, UnionForm):
            # arbitrarily many contents, possibly with missing fields
            for sublayout in layout.contents:
                if not any(compatible(subform, sublayout) for subform in form.contents):
                    return False
            return True

    elif isinstance(form, UnionForm):
        for subform in form.contents:
            if compatible(subform, layout):
                return True
        else:
            return False

    # handle other cases that come up here...

    else:
        return False


def _unproject_layout(form, layout, length, nplike):
    if layout is None:
        # construct the "minimum necessary" layout
        # maintaining length constraints if there are any, 0 otherwise

        if isinstance(form, EmptyForm):
            return EmptyArray(parameters=form.parameters)

        elif isinstance(form, NumpyForm):
            return NumpyArray(
                dummy_buffer(
                    (length,) + form.inner_shape,
                    ak.types.numpytype.primitive_to_dtype(form.primitive),
                    nplike,
                ),
                parameters=form.parameters,
            )

        elif isinstance(form, BitMaskedForm):
            return BitMaskedArray(
                index_of[form.mask](int(math.ceil(length / 8.0)), nplike),
                _unproject_layout(form.content, None, length, nplike),
                form.valid_when,
                length,
                form.lsb_order,
                parameters=form.parameters,
            )

        elif isinstance(form, ByteMaskedForm):
            return ByteMaskedArray(
                index_of[form.mask](length, nplike),
                _unproject_layout(form.content, None, length, nplike),
                form.valid_when,
                parameters=form.parameters,
            )

        elif isinstance(form, IndexedForm):
            return IndexedArray(
                index_of[form.index](length, nplike),
                _unproject_layout(form.content, None, 0, nplike),
                parameters=form.parameters,
            )

        elif isinstance(form, IndexedOptionForm):
            return IndexedOptionArray(
                index_of[form.index](length, nplike),
                _unproject_layout(form.content, None, 0, nplike),
                parameters=form.parameters,
            )

        elif isinstance(form, ListForm):
            return ListArray(
                index_of[form.starts](length, nplike),
                index_of[form.stops](length, nplike),
                _unproject_layout(form.content, None, 0, nplike),
                parameters=form.parameters,
            )

        elif isinstance(form, ListOffsetForm):
            return ListOffsetArray(
                index_of[form.offsets](length + 1, nplike),
                _unproject_layout(form.content, None, 0, nplike),
                parameters=form.parameters,
            )

        elif isinstance(form, RegularForm):
            return RegularArray(
                _unproject_layout(form.content, None, length * form.size, nplike),
                form.size,
                length,
                parameters=form.parameters,
            )

        elif isinstance(form, UnmaskedForm):
            return UnmaskedArray(
                _unproject_layout(form.content, None, length, nplike),
                parameters=form.parameters,
            )

        elif isinstance(form, RecordForm):
            return RecordArray(
                [
                    _unproject_layout(content, None, length, nplike)
                    for content in form.contents
                ],
                None if form.is_tuple else form.fields,
                length,
                parameters=form.parameters,
            )

        elif isinstance(form, UnionForm):
            return UnionArray(
                index_of[form.tags](length, nplike),
                index_of[form.index](length, nplike),
                [
                    _unproject_layout(content, None, 0, nplike)
                    for content in form.contents
                ],
                parameters=form.parameters,
            )

        else:
            raise AssertionError(f"unrecognized Form type: {type(form)}")

    elif isinstance(layout, Content) and type(form) is layout.form_cls:
        # pass on this layout node, allowing for descendants to be missing

        if isinstance(form, (EmptyForm, NumpyForm)):
            # 0 contents
            return layout

        elif isinstance(
            form,
            (
                BitMaskedForm,
                ByteMaskedForm,
                IndexedForm,
                IndexedOptionForm,
                ListForm,
                ListOffsetForm,
                RegularForm,
                UnmaskedForm,
            ),
        ):
            # 1 content
            return layout.copy(
                content=_unproject_layout(
                    form.content, layout.content, layout.content.length, nplike
                )
            )

        elif isinstance(form, RecordForm):
            # arbitrarily many contents, possibly with missing fields
            contents = []
            for field in form.fields:
                if layout.has_field(field):
                    layout_content = layout.content(field)
                    contents.append(
                        _unproject_layout(
                            form.content(field),
                            layout_content,
                            layout_content.length,
                            nplike,
                        )
                    )
                else:
                    contents.append(
                        _unproject_layout(form.content(field), None, length, nplike)
                    )

            return RecordArray(
                contents,
                None if form.is_tuple else form.fields,
                length,
                parameters=form.parameters,
            )

        elif isinstance(form, UnionForm):
            # arbitrarily many contents, possibly with missing fields
            available = dict(enumerate(layout.contents))

            newtags = nplike.empty_like(layout.tags.data)
            contents = []
            for newtag, subform in enumerate(form.contents):
                for oldtag, sublayout in available.items():
                    if compatible(subform, sublayout):
                        contents.append(sublayout)
                        newtags[layout.tags.data == oldtag] = newtag
                        del available[oldtag]
                        break
                else:
                    contents.append(_unproject_layout(subform, None, 0, nplike))

            return UnionArray(
                ak.index.Index8(newtags),
                layout.index,
                contents,
                parameters=form.parameters,
            )

        else:
            raise AssertionError(f"unrecognized Form type: {type(form)}")

    elif isinstance(form, UnionForm):
        newtags, newindex = None, None
        contents = []
        for newtag, subform in enumerate(form.contents):
            if compatible(subform, layout):
                contents.append(layout)
                newtags = nplike.full(layout.length, newtag, dtype=np.int8)
                newindex = nplike.arange(layout.length, dtype=dtype_of[form.index])
            else:
                contents.append(_unproject_layout(subform, None, 0, nplike))

        assert newtags is not None and newindex is not None
        return UnionArray(
            ak.index.Index8(newtags),
            ak.index.Index(newindex),
            contents,
            parameters=form.parameters,
        )

    # handle other cases that come up here...

    else:
        raise AssertionError(f"unexpected combination: {type(form)} and {type(layout)}")


def unproject_layout(form: Form, layout: Content) -> Content:
    return _unproject_layout(form, layout, layout.length, nplike_of(layout))
