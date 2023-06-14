from __future__ import annotations

from awkward.forms.form import Form


def set_form_keys(form: Form, *, key: str) -> Form:
    """Recursive function to apply key labels to `form`.

    Parameters
    ----------
    form : awkward.forms.form.Form
        Awkward Array form object to mutate.
    key : str
        Label to apply. If recursion is triggered by passing in a
        Record Form, the key is used as a prefix for a specific
        field.

    Returns
    -------
    awkward.forms.form.Form
        Mutated Form object.

    """

    # If the form is a record we need to loop over all fields in the
    # record and set form that include the field name; this will keep
    # recursing as well.
    if form.is_record:
        for field in form.fields:
            full_key = f"{key}.{field}"
            set_form_keys(form.content(field), key=full_key)

    # If the form is a list (e.g. ListOffsetArray) we append a
    # __list__ suffix to notify the optimization pass that we only
    # touched the offsets and not the data buffer for this kind of
    # identified form; keep recursing
    elif form.is_list:
        form.form_key = f"{key}.__list__"
        set_form_keys(form.content, key=key)

    # NumPy like array is easy
    elif form.is_numpy:
        form.form_key = key

    # Anything else grab the content and keep recursing
    else:
        set_form_keys(form.content, key=key)

    # Return the now mutated Form object.
    return form
