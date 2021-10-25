"""src/talus_utils/dataframe.py module."""

import functools

from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd

from .utils import override_args, override_kwargs


def copy(func: Callable[..., Any]) -> Callable[..., Any]:
    """Create a deep copy of a given pandas DataFrame and substitute it in the arguments.

    Parameters
    ----------
    func: Callable[..., Any] :
        The input function.

    Returns
    -------
    Callable[..., Any]
        The wrapped function.
    """

    @functools.wraps(func)
    def wrapped_func(*args: str, **kwargs: str) -> Any:
        """Substitute the arguments that are pandas DataFrames with a deep copy.

        Parameters
        ----------
        args :
            The arguments to the wrapped function.
        kwargs :
            The keyword arguments to the wrapped function.

        Returns
        -------
        Any
            The return value of the function it wraps.

        """
        apply_func = lambda df: df.copy(deep=True)
        filter_func = lambda arg: type(arg) == pd.DataFrame
        args = override_args(args=args, func=apply_func, filter=filter_func)
        kwargs = override_kwargs(kwargs=kwargs, func=apply_func, filter=filter_func)
        return_value = func(*args, **kwargs)
        return return_value

    return wrapped_func


def dropna(*pd_args: Union[int, str], **pd_kwargs: str) -> Callable[..., Any]:
    """Drop NaN values in a pandas DataFrame argument.

    Parameters
    ----------
    pd_args :
        The arguments to the wrapped function.
    pd_kwargs :
        The keyword arguments to the wrapped function.

    Returns
    -------
    Callable[..., Any]
        The wrapped function.

    """

    def dropna_wrap(func: Callable[..., Any]) -> Callable[..., Any]:
        """Drop NaN values in a pandas DataFrame argument.

        Parameters
        ----------
        func: Callable[..., Any] :
            The input function.

        Returns
        -------
        Callable[..., Any]
            The wrapped function.

        """

        @functools.wraps(func)
        def wrapped_func(*args: str, **kwargs: str) -> Any:
            apply_func = lambda df: df.dropna(*pd_args, **pd_kwargs)
            filter_func = lambda arg: type(arg) == pd.DataFrame
            args = override_args(args=args, func=apply_func, filter=filter_func)
            kwargs = override_kwargs(kwargs=kwargs, func=apply_func, filter=filter_func)
            return_value = func(*args, **kwargs)
            return return_value

        return wrapped_func

    return dropna_wrap


def log_scaling(
    log_function: Callable[..., Any] = np.log10, filter_outliers: bool = True
) -> Callable[..., Any]:
    """Apply a log scale to a given pandas DataFrame argument.

    Parameters
    ----------
    log_function : Callable[..., Any]
        The logarithm function to apply. (Default value = np.log10).
    filter_outliers : bool
        If False, set all values below 1 to 1 to ensure np.log works. (Default value = True).

    Returns
    -------
    Callable[..., Any]
        The wrapped function.

    """

    def log_scaling_wrap(func: Callable[..., Any]) -> Callable[..., Any]:
        """Apply a log scale to a given pandas DataFrame argument.

        Parameters
        ----------
        func: Callable[..., Any] :
            The input function.

        Returns
        -------
        Callable[..., Any]
            The wrapped function.

        """

        @functools.wraps(func)
        def wrapped_func(*args: str, **kwargs: str) -> Any:
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


def pivot_table(*pd_args: str, **pd_kwargs: str) -> Callable[..., Any]:
    """Apply a pivot to a pandas DataFrame argument.

    Parameters
    ----------
    pd_args :
        The arguments to the wrapped function.
    pd_kwargs :
        The keyword arguments to the wrapped function.

    Returns
    -------
    Callable[..., Any]
        The wrapped function.

    """

    def pivot_table_wrap(func: Callable[..., Any]) -> Callable[..., Any]:
        """Apply a pivot to a given pandas DataFrame argument.

        Parameters
        ----------
        func: Callable[..., Any] :
            The input function.

        Returns
        -------
        Callable[..., Any]
            The wrapped function.

        """

        @functools.wraps(func)
        def wrapped_func(*args: str, **kwargs: str) -> Any:
            apply_func = lambda df: df.pivot_table(*pd_args, **pd_kwargs)
            filter_func = lambda arg: type(arg) == pd.DataFrame
            args = override_args(args=args, func=apply_func, filter=filter_func)
            kwargs = override_kwargs(kwargs=kwargs, func=apply_func, filter=filter_func)
            return_value = func(*args, **kwargs)
            return return_value

        return wrapped_func

    return pivot_table_wrap


def normalize(how: str) -> Callable[..., Any]:
    """Apply a row or column normalization to a pandas DataFrame argument.

    Parameters
    ----------
    how : str
        The normalization method to apply. Can be one of {'row', 'colum', 'minmax', 'median_column'}.
        'row': Normalize each row to the range [0, 1].
        'colum': Normalize each column to the range [0, 1].
        'minmax': Apply a min-max normalization.
        'median_column': Scale each column by subtracting the median value.

    Returns
    -------
    Callable[..., Any]
        The wrapped function.

    """

    def normalize_wrap(func: Callable[..., Any]) -> Callable[..., Any]:
        """Apply a row or column normalization to a pandas DataFrame argument.

        Parameters
        ----------
        func: Callable[..., Any] :
            The input function.

        Returns
        -------
        Callable[..., Any]
            The wrapped function.

        """

        @functools.wraps(func)
        def wrapped_func(*args: str, **kwargs: str) -> Any:
            if how.lower() in set(["row", "r"]):
                apply_func = lambda df: df.apply(lambda x: x / x.sum(), axis=1)
            elif how.lower() in set(["column", "col", "c"]):
                apply_func = lambda df: df.apply(lambda x: x / x.sum(), axis=0)
            elif how.lower() in set(["minmax", "min-max", "min_max"]):
                apply_func = lambda df: (df - df.min()) / (df.max() - df.min())
            elif how.lower() in set(["median_column", "median_col"]):
                apply_func = lambda df: df - df.median(axis=0)
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
    how: str,
    use_absolute_values: Optional[bool] = False,
    sort_ascending: Optional[bool] = False,
) -> Callable[..., Any]:
    """Reindex a pandas DataFrame argument.

    Parameters
    ----------
    how : str
        The reindexing method to apply. Can be one of {'min', 'max', 'median', 'mean', 'sum'}.
    use_absolute_values : bool
        If True, use absolute values of the row values. (Default value = False).
    sort_ascending : bool
        Whether to sort the index in ascending order. (Default value = False).

    Returns
    -------
    Callable[..., Any]
        The wrapped function.

    """

    def reindex_wrap(func: Callable[..., Any]) -> Callable[..., Any]:
        """Reindexe a pandas DataFrame argument.

        Parameters
        ----------
        func: Callable[..., Any] :
            The input function.

        Returns
        -------
        Callable[..., Any]
            The wrapped function.

        """

        @functools.wraps(func)
        def wrapped_func(*args: str, **kwargs: str) -> Any:
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


def explode_column(
    df: pd.DataFrame,
    column: str,
    sep: Optional[str] = ";",
    ignore_index: Optional[bool] = False,
) -> pd.DataFrame:
    """Explode a column in a given pandas DataFrame argument.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    column : str
        The column to explode.
    sep : Optional[str], optional
        The string to use to separate values in the resulting DataFrame. (Default value = None).
    ignore_index : Optional[bool], optional
        If True, the resulting index will be labeled 0, 1, …, n - 1. (Default value = False).

    Returns
    -------
    pd.DataFrame
        A DataFrame with the given column exploded.
    """
    return df.assign(
        **{
            column: df[column].apply(
                lambda row: row.split(sep) if isinstance(row, str) else row
            )
        }
    ).explode(column=column, ignore_index=ignore_index)


def explode(
    column: str,
    ignore_index: Optional[bool] = False,
    sep: Optional[str] = None,
) -> Callable[..., Any]:
    """Explode a column in a given pandas DataFrame argument.

    Parameters
    ----------
    column : str
        The column to explode.
    ignore_index : Optional[bool]
        If True, the resulting index will be labeled 0, 1, …, n - 1. (Default value = False).
    sep : Optional[str]
        The string to use to separate values in the resulting DataFrame. (Default value = None).

    Returns
    -------
    Callable[..., Any]
        The wrapped function.

    """

    def explode_wrap(func: Callable[..., Any]) -> Callable[..., Any]:
        """Explode a column in a given pandas DataFrame argument.

        Parameters
        ----------
        func: Callable[..., Any] :
            The input function.

        Returns
        -------
        Callable[..., Any]
            The wrapped function.

        """

        @functools.wraps(func)
        def wrapped_func(*args: str, **kwargs: str) -> Any:
            if sep:
                apply_func = lambda df: explode_column(
                    df=df, column=column, sep=sep, ignore_index=ignore_index
                )
            else:
                apply_func = lambda df: df.explode(
                    column=column, ignore_index=ignore_index
                )
            filter_func = lambda arg: type(arg) == pd.DataFrame
            args = override_args(args=args, func=apply_func, filter=filter_func)
            kwargs = override_kwargs(kwargs=kwargs, func=apply_func, filter=filter_func)
            return_value = func(*args, **kwargs)
            return return_value

        return wrapped_func

    return explode_wrap


def update_column(column: str, update_func: Callable[..., Any]) -> Callable[..., Any]:
    """Apply a given function to a column in a given pandas DataFrame argument.

    Parameters
    ----------
    column : str
        The name of the column to update.
    update_func : Callable[..., Any]
        The function to apply to the column.

    Returns
    -------
    Callable[..., Any]
        The wrapped function.

    """

    def update_column_wrap(func: Callable[..., Any]) -> Callable[..., Any]:
        """Apply a given function to a column in a given pandas DataFrame argument.

        Parameters
        ----------
        func: Callable[..., Any] :
            The input function.

        Returns
        -------
        Callable[..., Any]
            The wrapped function.

        """

        @functools.wraps(func)
        def wrapped_func(*args: str, **kwargs: str) -> Any:
            apply_func = lambda df: df.assign(**{column: df[column].apply(update_func)})
            filter_func = lambda arg: type(arg) == pd.DataFrame
            args = override_args(args=args, func=apply_func, filter=filter_func)
            kwargs = override_kwargs(kwargs=kwargs, func=apply_func, filter=filter_func)
            return_value = func(*args, **kwargs)
            return return_value

        return wrapped_func

    return update_column_wrap
