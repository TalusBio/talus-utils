"""tests/test_algorithms.py module."""
from pathlib import Path

import pandas as pd

from pandas.testing import assert_frame_equal

from talus_utils import algorithms


DATA_DIR = Path(__file__).resolve().parent.joinpath("data")


def test_subcellular_enrichment_scores() -> None:
    """Test the subcellular_enrichment_scores function."""
    df_expected = pd.read_csv(DATA_DIR.joinpath("subcellular_enrichment_scores.csv"))
    df_expected = df_expected.set_index("Main location")

    df_input = pd.read_parquet(DATA_DIR.joinpath("proteins_with_locations.parquet"))
    df_expected_fractions = pd.read_parquet(
        DATA_DIR.joinpath("expected_fractions_of_locations.parquet")
    )

    df_actual = algorithms.subcellular_enrichment_scores(
        proteins_with_locations=df_input,
        expected_fractions_of_locations=df_expected_fractions,
    )

    assert_frame_equal(df_actual, df_expected)
