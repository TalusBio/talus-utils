"""tests/test_algorithms.py"""
from pathlib import Path

import pandas as pd

from pandas.testing import assert_frame_equal

from talus_utils import algorithms


DATA_DIR = Path(__file__).resolve().parent.joinpath("data")


def test_get_outlier_peptide_intensities() -> None:
    """Test the get_outlier_peptide_intensities function."""
    df_expected = pd.read_csv(DATA_DIR.joinpath("quant_peptides_outliers.csv"))
    df_expected = df_expected.set_index("Peptide")

    df_input = pd.read_csv(DATA_DIR.joinpath("quant_peptides.txt"), sep="\t")
    df_input = df_input.drop(["Protein", "numFragments"], axis=1)
    df_input = df_input.drop_duplicates(subset="Peptide")
    df_input = df_input.set_index(["Peptide"])

    df_actual = algorithms.get_outlier_peptide_intensities(peptide_intensities=df_input)

    assert_frame_equal(df_actual, df_expected)


def test_get_outlier_peptide_intensities_above_below() -> None:
    """Test the get_outlier_peptide_intensities function with split_above_below."""
    df_expected_above = pd.read_csv(
        DATA_DIR.joinpath("quant_peptides_outliers_above_mean.csv")
    )
    df_expected_above = df_expected_above.set_index("Peptide")

    df_expected_below = pd.read_csv(
        DATA_DIR.joinpath("quant_peptides_outliers_below_mean.csv")
    )
    df_expected_below = df_expected_below.set_index("Peptide")

    df_input = pd.read_csv(DATA_DIR.joinpath("quant_peptides.txt"), sep="\t")
    df_input = df_input.drop(["Protein", "numFragments"], axis=1)
    df_input = df_input.drop_duplicates(subset="Peptide")
    df_input = df_input.set_index(["Peptide"])

    df_actual_above, df_actual_below = algorithms.get_outlier_peptide_intensities(
        peptide_intensities=df_input, split_above_below=True
    )

    assert_frame_equal(df_actual_above, df_expected_above)
    assert_frame_equal(df_actual_below, df_expected_below)


def test_get_hits_for_proteins() -> None:
    """Test the get_hits_for_proteins function."""
    # using min_peptides = 1 for testing purposes
    min_peptides = 1
    df_expected = pd.read_csv(DATA_DIR.joinpath("hits_for_proteins.csv"))
    df_expected = df_expected.set_index("Protein")

    df_outlier_peptides = pd.read_csv(DATA_DIR.joinpath("quant_peptides_outliers.csv"))
    df_outlier_peptides = df_outlier_peptides.set_index("Peptide")

    df_peptides = pd.read_csv(DATA_DIR.joinpath("quant_peptides.txt"), sep="\t")
    df_peptides = df_peptides[["Peptide", "Protein"]]
    df_peptides["NumPeptides"] = df_peptides.groupby("Protein").transform("count")
    df_peptides = df_peptides[df_peptides["NumPeptides"] >= min_peptides]

    df_actual = algorithms.get_hits_for_proteins(
        outlier_peptide_intensities=df_outlier_peptides,
        peptide_df=df_peptides,
    )

    assert_frame_equal(df_actual, df_expected)


def test_hit_selection() -> None:
    """Test the hit_selection function."""
    df_expected = pd.read_csv(DATA_DIR.joinpath("hits_for_proteins.csv"))
    df_expected = df_expected.set_index("Protein")

    df_input = pd.read_csv(DATA_DIR.joinpath("quant_peptides.txt"), sep="\t")
    df_input = df_input.drop(["numFragments"], axis=1)

    df_actual = algorithms.hit_selection(
        peptide_df=df_input,
        min_peptides=1,
    )

    assert_frame_equal(df_actual, df_expected)


def test_hit_selection_above_below() -> None:
    """Test the hit_selection function with split_above_below."""
    df_expected_above = pd.read_csv(
        DATA_DIR.joinpath("hits_for_proteins_above_mean.csv")
    )
    df_expected_above = df_expected_above.set_index("Protein")

    df_expected_below = pd.read_csv(
        DATA_DIR.joinpath("hits_for_proteins_below_mean.csv")
    )
    df_expected_below = df_expected_below.set_index("Protein")

    df_input = pd.read_csv(DATA_DIR.joinpath("quant_peptides.txt"), sep="\t")
    df_input = df_input.drop(["numFragments"], axis=1)

    df_actual_above, df_actual_below = algorithms.hit_selection(
        peptide_df=df_input,
        min_peptides=1,
        split_above_below=True,
    )

    assert_frame_equal(df_actual_above, df_expected_above)
    assert_frame_equal(df_actual_below, df_expected_below)
