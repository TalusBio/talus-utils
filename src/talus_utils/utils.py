"""src/talus_utils/utils.py module."""

from typing import Callable, Dict, List, Tuple


def override_args(
    args: Tuple, func: Callable, filter: Callable = lambda _: True
) -> List:
    """Override the args of a function given a function to apply and an optional filter.

    Parameters
    ----------
    args : Tuple
        The function args input.
    func : Callable
        A function to apply on the args.
    filter : Callable
        An optional filter to apply the function only on some args. (Default value = lambda _: True).

    Returns
    -------
    List
        The changed args as a List.

    """
    return [func(arg) if filter(arg) else arg for arg in list(args)]


def override_kwargs(
    kwargs: Tuple, func: Callable, filter: Callable = lambda _: True
) -> Dict:
    """Override the kwargs of a function given a function to apply and an optional filter.

    Parameters
    ----------
    kwargs : Tuple
        The function kwargs input.
    func : Callable
        A function to apply on the kwargs.
    filter : Callable
        An optional filter to apply the function only on some kwargs. (Default value = lambda _: True).

    Returns
    -------
    Dict
        The changed kwargs as a Dict.

    """
    return {
        key: func(value) if filter(value) else value for key, value in kwargs.items()
    }
