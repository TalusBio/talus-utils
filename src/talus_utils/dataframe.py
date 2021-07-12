"""src/talus_utils/dataframe.py"""

import functools

from typing import Any, Callable

import numpy as np
import pandas as pd

from .utils import override_args, override_kwargs


def copy(func: Callable) -> Callable:
    """Function decorator that creates a deep copy of a given pandas DataFrame
    and substitutes it in the arguments.
    """

    @functools.wraps(func)
    def wrapped_func(*args, **kwargs) -> Any:
        """A function wrapper that substitutes the arguments that are pandas DataFrames
        with a deep copy.

        Returns:
            Any: The return value of the function it wraps.
        """
        apply_func = lambda df: df.copy(deep=True)
        filter_func = lambda arg: type(arg) == pd.DataFrame
        args = override_args(args=args, func=apply_func, filter=filter_func)
        kwargs = override_kwargs(kwargs=kwargs, func=apply_func, filter=filter_func)
        return_value = func(*args, **kwargs)
        return return_value

    return wrapped_func


def dropna(*pd_args, **pd_kwargs) -> Callable:
    """Function decorator that drops NaN values in a pandas DataFrame argument.

    Returns:
        Callable: The wrapped function.
    """

    def dropna_wrap(func: Callable) -> Callable:
        """Function decorator that drops NaN values in a pandas DataFrame argument."""

        @functools.wraps(func)
        def wrapped_func(*args, **kwargs) -> Any:
            apply_func = lambda df: df.dropna(*pd_args, **pd_kwargs)
            filter_func = lambda arg: type(arg) == pd.DataFrame
            args = override_args(args=args, func=apply_func, filter=filter_func)
            kwargs = override_kwargs(kwargs=kwargs, func=apply_func, filter=filter_func)
            return_value = func(*args, **kwargs)
            return return_value

        return wrapped_func

    return dropna_wrap


def log_scaling(
    log_function: Callable = np.log10, filter_outliers: bool = True
) -> Callable:
    """Function decorator that applies a log scale to a given pandas DataFrame argument.

    Args:
        log_function (Callable, optional): The logarithm function to apply. Defaults to np.log10.
        filter_outliers (bool, optional): If False, set all values below 1 to 1 to ensure np.log works. Defaults to True.

    Returns:
        Callable: The wrapped function.
    """

    def log_scaling_wrap(func: Callable) -> Callable:
        """Function decorator that applies a log scale to a given pandas DataFrame argument."""

        @functools.wraps(func)
        def wrapped_func(*args, **kwargs) -> Any:
            if filter_outliers:
                apply_func = lambda df: log_function(df.where(df >= 1))
            else:
                apply_func = lambda df: log_function(df.mask(df < 1, 1))
            filter_func = lambda arg: type(arg) == pd.DataFrame
            args = override_args(args=args, func=apply_func, filter=filter_func)
            kwargs = override_kwargs(kwargs=kwargs, func=apply_func, filter=filter_func)
            return_value = func(*args, **kwargs)
            return return_value

        return wrapped_func

    return log_scaling_wrap


def pivot_table(*pd_args, **pd_kwargs) -> Callable:
    """Function decorator that applies a pivot to a pandas DataFrame argument.

    Returns:
        Callable: The wrapped function.
    """

    def pivot_table_wrap(func: Callable) -> Callable:
        """Function decorator that applies a pivot to a given pandas DataFrame argument."""

        @functools.wraps(func)
        def wrapped_func(*args, **kwargs) -> Any:
            apply_func = lambda df: df.pivot_table(*pd_args, **pd_kwargs)
            filter_func = lambda arg: type(arg) == pd.DataFrame
            args = override_args(args=args, func=apply_func, filter=filter_func)
            kwargs = override_kwargs(kwargs=kwargs, func=apply_func, filter=filter_func)
            return_value = func(*args, **kwargs)
            return return_value

        return wrapped_func

    return pivot_table_wrap


def normalize(how: str) -> Callable:
    """Function decorator that applies a row or column normalization to a pandas DataFrame argument.

    Args:
        how (str): The normalization method to apply. Can be one of {'row', 'colum', 'minmax'}.

    Raises:
        ValueError: If the given argument is not one of {'row', 'colum', 'minmax'}.

    Returns:
        Callable: The wrapped function.
    """

    def normalize_wrap(func: Callable) -> Callable:
        """Function decorator that applies a row or column normalization to a pandas DataFrame argument."""

        @functools.wraps(func)
        def wrapped_func(*args, **kwargs) -> Any:
            if how.lower() in set(["row", "r"]):
                apply_func = lambda df: df.apply(lambda x: x / x.sum(), axis=1)
            elif how.lower() in set(["column", "col", "c"]):
                apply_func = lambda df: df.apply(lambda x: x / x.sum(), axis=0)
            elif how.lower() in set(["minmax", "min-max", "min_max"]):
                apply_func = lambda df: (df - df.min()) / (df.max() - df.min())
            else:
                raise ValueError(
                    "Invalid input value for 'how'. Needs to be one of {'row', 'colum', 'minmax'}."
                )

            filter_func = lambda arg: type(arg) == pd.DataFrame
            args = override_args(args=args, func=apply_func, filter=filter_func)
            kwargs = override_kwargs(kwargs=kwargs, func=apply_func, filter=filter_func)
            return_value = func(*args, **kwargs)
            return return_value

        return wrapped_func

    return normalize_wrap


def sort_row_values(
    how: str, use_absolute_values: bool = False, sort_ascending: bool = False
) -> Callable:
    """Function decorator that reindexes a pandas DataFrame argument.

    Args:
        how (str): The reindexing method to apply. Can be one of {'min', 'max', 'median', 'mean', 'sum'}.
        use_absolute_values (bool, optional): If True, use absolute values of the row values. Defaults to False.
        sort_ascending (bool, optional): Whether to sort the index in ascending order. Defaults to False.

    Raises:
        ValueError: If the given argument is not one of {'min', 'max', 'median', 'mean', 'sum'}.

    Returns:
        Callable: The wrapped function.
    """

    def reindex_wrap(func: Callable) -> Callable:
        """Function decorator that reindexes a pandas DataFrame argument."""

        @functools.wraps(func)
        def wrapped_func(*args, **kwargs) -> Any:
            if how.lower() == "min":
                if use_absolute_values:
                    apply_func = lambda df: df.reindex(
                        index=df.abs()
                        .min(axis=1)
                        .sort_values(ascending=sort_ascending)
                        .index
                    )
                else:
                    apply_func = lambda df: df.reindex(
                        index=df.min(axis=1).sort_values(ascending=sort_ascending).index
                    )
            elif how.lower() == "max":
                if use_absolute_values:
                    apply_func = lambda df: df.reindex(
                        index=df.abs()
                        .max(axis=1)
                        .sort_values(ascending=sort_ascending)
                        .index
                    )
                else:
                    apply_func = lambda df: df.reindex(
                        index=df.max(axis=1).sort_values(ascending=sort_ascending).index
                    )
            elif how.lower() == "median":
                if use_absolute_values:
                    apply_func = lambda df: df.reindex(
                        index=df.abs()
                        .median(axis=1)
                        .sort_values(ascending=sort_ascending)
                        .index
                    )
                else:
                    apply_func = lambda df: df.reindex(
                        index=df.median(axis=1)
                        .sort_values(ascending=sort_ascending)
                        .index
                    )
            elif how.lower() == "mean":
                if use_absolute_values:
                    apply_func = lambda df: df.reindex(
                        index=df.abs()
                        .mean(axis=1)
                        .sort_values(ascending=sort_ascending)
                        .index
                    )
                else:
                    apply_func = lambda df: df.reindex(
                        index=df.mean(axis=1)
                        .sort_values(ascending=sort_ascending)
                        .index
                    )
            elif how.lower() == "sum":
                if use_absolute_values:
                    apply_func = lambda df: df.reindex(
                        index=df.abs()
                        .sum(axis=1)
                        .sort_values(ascending=sort_ascending)
                        .index
                    )
                else:
                    apply_func = lambda df: df.reindex(
                        index=df.sum(axis=1).sort_values(ascending=sort_ascending).index
                    )
            else:
                raise ValueError(
                    "Invalid input value for 'how'. Needs to be one of {'min', 'max', 'median', 'mean', 'sum'}."
                )

            filter_func = lambda arg: type(arg) == pd.DataFrame
            args = override_args(args=args, func=apply_func, filter=filter_func)
            kwargs = override_kwargs(kwargs=kwargs, func=apply_func, filter=filter_func)
            return_value = func(*args, **kwargs)
            return return_value

        return wrapped_func

    return reindex_wrap
