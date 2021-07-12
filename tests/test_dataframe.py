"""tests/test_dataframe.py"""
import numpy as np
import pandas as pd
import pytest

from pandas.testing import assert_frame_equal

from talus_utils import dataframe


def assert_frame_not_equal(*args, **kwargs):
    """The inverse of pandas.testing.assert_frame_equal."""
    try:
        assert_frame_equal(*args, **kwargs)
    except AssertionError:
        # frames are not equal
        pass
    else:
        # frames are equal
        raise AssertionError


def dummy_function_change_df_column(df: pd.DataFrame) -> None:
    """A dummy function that changes the input df's column."""
    df["test"] = "dummy_value"


def dummy_function(df: pd.DataFrame) -> pd.DataFrame:
    """A dummy function that just returns the input DataFrame."""
    return df


def test_copy_df_in_args() -> None:
    """Tests the copy decorator with the df in the args."""
    df_input = pd.DataFrame([{"test": "a", "test2": "b"}, {"test": "c", "test2": "d"}])
    df_expected_to_change = df_input.copy(deep=True)
    df_expected_not_to_change = df_input.copy(deep=True)

    dummy_function_change_df_column(df_expected_to_change)
    assert_frame_not_equal(df_expected_to_change, df_input)

    dataframe.copy(dummy_function_change_df_column)(df_expected_not_to_change)
    assert_frame_equal(df_expected_not_to_change, df_input)


def test_copy_df_in_kwargs() -> None:
    """Tests the copy decorator with the df in the kwargs."""
    df_input = pd.DataFrame([{"test": "a", "test2": "b"}, {"test": "c", "test2": "d"}])
    df_expected_to_change = df_input.copy(deep=True)
    df_expected_not_to_change = df_input.copy(deep=True)

    dummy_function_change_df_column(df=df_expected_to_change)
    assert_frame_not_equal(df_expected_to_change, df_input)

    dataframe.copy(dummy_function_change_df_column)(df=df_expected_not_to_change)
    assert_frame_equal(df_expected_not_to_change, df_input)


def test_dropna() -> None:
    """Tests the dropna decorator."""
    df_input = pd.DataFrame(
        [{"test": "a", "test2": np.nan}, {"test": "c", "test2": "d"}]
    )
    df_expected = df_input.dropna()

    df_actual = dataframe.dropna()(dummy_function)(df_input)
    assert_frame_equal(df_actual, df_expected)


def test_dropna_column() -> None:
    """Tests the dropna decorator by dropping a column."""
    df_input = pd.DataFrame(
        [{"test": "a", "test2": np.nan}, {"test": "c", "test2": "d"}]
    )
    df_expected = pd.DataFrame([{"test": "a"}, {"test": "c"}])

    df_actual = dataframe.dropna(axis=1)(dummy_function)(df_input)
    assert_frame_equal(df_actual, df_expected)


def test_log_scaling() -> None:
    """Tests the log_scaling decorator."""
    df_input = pd.DataFrame(np.random.rand(5, 5) * 100)
    df_expected = np.log10(df_input)

    df_actual = dataframe.log_scaling()(dummy_function)(df_input)
    assert_frame_equal(df_actual, df_expected)


def test_log_scaling_custom_log() -> None:
    """Tests the log_scaling decorator with a custom log function (np.log2)."""
    df_input = pd.DataFrame(np.random.rand(5, 5) * 100)
    df_expected = np.log2(df_input.where(df_input >= 1))

    df_actual = dataframe.log_scaling(log_function=np.log2)(dummy_function)(df_input)
    assert_frame_equal(df_actual, df_expected)


def test_log_scaling_with_zeros() -> None:
    """Tests the log_scaling decorator with zeros (should be filtered out and set to NaN)."""
    df_input = pd.DataFrame([{"test": 25, "test2": 0}, {"test": 26, "test2": 42}])
    df_expected = np.log10(df_input.where(df_input >= 1))

    df_actual = dataframe.log_scaling()(dummy_function)(df_input)
    assert_frame_equal(df_actual, df_expected)


def test_log_scaling_with_zeros_no_filter_outliers() -> None:
    """Tests the log_scaling decorator with zeros and without filtering outliers."""
    df_input = pd.DataFrame([{"test": 25, "test2": 0}, {"test": 26, "test2": 42}])
    df_expected = np.log10(df_input.mask(df_input < 1, 1))

    df_actual = dataframe.log_scaling(filter_outliers=False)(dummy_function)(df_input)
    assert_frame_equal(df_actual, df_expected)


def test_pivot_table() -> None:
    """Tests the test_pivot_table decorator."""
    df_input = pd.DataFrame(
        [
            {"index": "a", "column": "b", "value": 1},
            {"index": "c", "column": "d", "value": 2},
        ]
    )
    df_expected = df_input.pivot_table(index="index", columns="column", values="value")

    df_actual = dataframe.pivot_table(index="index", columns="column", values="value")(
        dummy_function
    )(df_input)
    assert_frame_equal(df_actual, df_expected)


def test_normalize_value_error() -> None:
    """Tests the normalize decorator with a value error."""
    df_input = pd.DataFrame([{"test": "a", "test2": "b"}, {"test": "c", "test2": "d"}])

    with pytest.raises(ValueError):
        _ = dataframe.normalize(how="nonexisting")(dummy_function)(df_input)


def test_normalize_minmax() -> None:
    """Tests the normalize decorator with how='minmax'."""
    df_input = pd.DataFrame(np.random.rand(5, 5) * 100)
    df_expected = (df_input - df_input.min()) / (df_input.max() - df_input.min())

    df_actual = dataframe.normalize(how="minmax")(dummy_function)(df_input)
    assert_frame_equal(df_actual, df_expected)


def test_normalize_row() -> None:
    """Tests the normalize decorator with how='row'."""
    df_input = pd.DataFrame(np.random.rand(5, 5) * 100)
    df_expected = df_input.apply(lambda x: x / x.sum(), axis=1)

    df_actual = dataframe.normalize(how="row")(dummy_function)(df_input)
    assert_frame_equal(df_actual, df_expected)


def test_normalize_column() -> None:
    """Tests the normalize decorator with how='column'."""
    df_input = pd.DataFrame(np.random.rand(5, 5) * 100)
    df_expected = df_input.apply(lambda x: x / x.sum(), axis=0)

    df_actual = dataframe.normalize(how="column")(dummy_function)(df_input)
    assert_frame_equal(df_actual, df_expected)


def test_sort_row_values_value_error() -> None:
    """Tests the sort_by decorator with a value error."""
    df_input = pd.DataFrame([{"test": "a", "test2": "b"}, {"test": "c", "test2": "d"}])

    with pytest.raises(ValueError):
        _ = dataframe.sort_row_values(how="nonexisting")(dummy_function)(df_input)


def test_sort_row_values_max() -> None:
    """Tests the sort_row_values decorator with 'max'."""
    df_input = pd.DataFrame(np.random.rand(5, 5) * 100)
    df_abs_expected = df_input.reindex(
        index=df_input.abs().max(axis=1).sort_values(ascending=True).index
    )

    df_actual = dataframe.sort_row_values(
        how="max", use_absolute_values=True, sort_ascending=True
    )(dummy_function)(df_input)
    assert_frame_equal(df_actual, df_abs_expected)

    df_expected = df_input.reindex(
        index=df_input.max(axis=1).sort_values(ascending=True).index
    )

    df_actual = dataframe.sort_row_values(how="max", sort_ascending=True)(
        dummy_function
    )(df_input)
    assert_frame_equal(df_actual, df_expected)

    df_desc_expected = df_input.reindex(
        index=df_input.max(axis=1).sort_values(ascending=False).index
    )

    df_actual = dataframe.sort_row_values(how="max")(dummy_function)(df_input)
    assert_frame_equal(df_actual, df_desc_expected)


def test_sort_row_values_min() -> None:
    """Tests the sort_row_values decorator with 'min'."""
    df_input = pd.DataFrame(np.random.rand(5, 5) * 100)
    df_abs_expected = df_input.reindex(
        index=df_input.abs().min(axis=1).sort_values(ascending=True).index
    )

    df_actual = dataframe.sort_row_values(
        how="min", use_absolute_values=True, sort_ascending=True
    )(dummy_function)(df_input)
    assert_frame_equal(df_actual, df_abs_expected)

    df_expected = df_input.reindex(
        index=df_input.min(axis=1).sort_values(ascending=True).index
    )

    df_actual = dataframe.sort_row_values(how="min", sort_ascending=True)(
        dummy_function
    )(df_input)
    assert_frame_equal(df_actual, df_expected)

    df_desc_expected = df_input.reindex(
        index=df_input.min(axis=1).sort_values(ascending=False).index
    )

    df_actual = dataframe.sort_row_values(how="min")(dummy_function)(df_input)
    assert_frame_equal(df_actual, df_desc_expected)


def test_sort_row_values_median() -> None:
    """Tests the sort_row_values decorator with 'median'."""
    df_input = pd.DataFrame(np.random.rand(5, 5) * 100)
    df_abs_expected = df_input.reindex(
        index=df_input.abs().median(axis=1).sort_values(ascending=True).index
    )

    df_actual = dataframe.sort_row_values(
        how="median", use_absolute_values=True, sort_ascending=True
    )(dummy_function)(df_input)
    assert_frame_equal(df_actual, df_abs_expected)

    df_expected = df_input.reindex(
        index=df_input.median(axis=1).sort_values(ascending=True).index
    )

    df_actual = dataframe.sort_row_values(how="median", sort_ascending=True)(
        dummy_function
    )(df_input)
    assert_frame_equal(df_actual, df_expected)

    df_desc_expected = df_input.reindex(
        index=df_input.median(axis=1).sort_values(ascending=False).index
    )

    df_actual = dataframe.sort_row_values(how="median")(dummy_function)(df_input)
    assert_frame_equal(df_actual, df_desc_expected)


def test_sort_row_values_mean() -> None:
    """Tests the sort_row_values decorator with 'mean'."""
    df_input = pd.DataFrame(np.random.rand(5, 5) * 100)
    df_abs_expected = df_input.reindex(
        index=df_input.abs().mean(axis=1).sort_values(ascending=True).index
    )

    df_actual = dataframe.sort_row_values(
        how="mean", use_absolute_values=True, sort_ascending=True
    )(dummy_function)(df_input)
    assert_frame_equal(df_actual, df_abs_expected)

    df_expected = df_input.reindex(
        index=df_input.mean(axis=1).sort_values(ascending=True).index
    )

    df_actual = dataframe.sort_row_values(how="mean", sort_ascending=True)(
        dummy_function
    )(df_input)
    assert_frame_equal(df_actual, df_expected)

    df_desc_expected = df_input.reindex(
        index=df_input.mean(axis=1).sort_values(ascending=False).index
    )

    df_actual = dataframe.sort_row_values(how="mean")(dummy_function)(df_input)
    assert_frame_equal(df_actual, df_desc_expected)


def test_sort_row_values_sum() -> None:
    """Tests the sort_row_values decorator with 'sum'."""
    df_input = pd.DataFrame(np.random.rand(5, 5) * 100)
    df_abs_expected = df_input.reindex(
        index=df_input.abs().sum(axis=1).sort_values(ascending=True).index
    )

    df_actual = dataframe.sort_row_values(
        how="sum", use_absolute_values=True, sort_ascending=True
    )(dummy_function)(df_input)
    assert_frame_equal(df_actual, df_abs_expected)

    df_expected = df_input.reindex(
        index=df_input.sum(axis=1).sort_values(ascending=True).index
    )

    df_actual = dataframe.sort_row_values(how="sum", sort_ascending=True)(
        dummy_function
    )(df_input)
    assert_frame_equal(df_actual, df_expected)

    df_desc_expected = df_input.reindex(
        index=df_input.sum(axis=1).sort_values(ascending=False).index
    )

    df_actual = dataframe.sort_row_values(how="sum")(dummy_function)(df_input)
    assert_frame_equal(df_actual, df_desc_expected)