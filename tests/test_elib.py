"""tests/test_elib.py module."""
from pathlib import Path

import pandas as pd

from deepdiff import DeepDiff

from talus_utils.elib import Elib


DATA_DIR = Path(__file__).resolve().parent.joinpath("data")

ELIB_FILE_KEY = DATA_DIR.joinpath("test.elib")
DF_EXPECTED = pd.read_csv(DATA_DIR.joinpath("select_peptidetoprotein.csv"))


def test_execute_sql_cursor() -> None:
    """Tests execute_sql using the cursor."""
    values_expected = [tuple(record) for record in DF_EXPECTED.to_records(index=False)]
    elib_conn = Elib(key_or_filename=ELIB_FILE_KEY)

    cursor = elib_conn.execute_sql(sql="SELECT * FROM peptidetoprotein;")
    values_actual = cursor.fetchall()
    assert (
        DeepDiff(
            values_actual,
            values_expected,
            ignore_string_type_changes=True,
            ignore_numeric_type_changes=True,
        )
        == {}
    )


def test_execute_sql_pandas() -> None:
    """Tests execute_sql using pandas."""
    elib_conn = Elib(key_or_filename=ELIB_FILE_KEY)

    df_actual = elib_conn.execute_sql(
        sql="SELECT * FROM peptidetoprotein;", use_pandas=True
    )
    assert type(df_actual) == pd.DataFrame
    pd.testing.assert_frame_equal(df_actual, DF_EXPECTED)
