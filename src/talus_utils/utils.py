"""src/talus_utils/utils.py"""

from typing import Callable, Dict, List, Tuple


def override_args(
    args: Tuple, func: Callable, filter: Callable = lambda _: True
) -> List:
    """Function to override the args of a function given a function to apply and an optional filter.

    Args:
        args (Tuple): The function args input.
        func (Callable): A function to apply on the args.
        filter (Callable, optional): An optional filter to apply the function only on some args. Defaults to lambda_:True.

    Returns:
        List: The changed args as a List.
    """
    return [func(arg) if filter(arg) else arg for arg in list(args)]


def override_kwargs(
    kwargs: Tuple, func: Callable, filter: Callable = lambda _: True
) -> Dict:
    """Function to override the kwargs of a function given a function to apply and an optional filter.

    Args:
        kwargs (Tuple): The function kwargs input.
        func (Callable): A function to apply on the kwargs.
        filter (Callable, optional): An optional filter to apply the function only on some kwargs. Defaults to lambda_:True.

    Returns:
        Dict: The changed kwargs as a Dict.
    """
    return {
        key: func(value) if filter(value) else value for key, value in kwargs.items()
    }
