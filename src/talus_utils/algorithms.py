"""src/talus_utils/algorithms.py module."""
from typing import Tuple, Union

import numpy as np
import pandas as pd

from . import dataframe as df_utils
from .constants import MAX_NAN_VALUES_HIT_SELECTION, MIN_PEPTIDES_HIT_SELECTION


def get_hits_for_proteins(
    outlier_peptide_intensities: pd.DataFrame,
    peptide_df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate the percentage of peptides that are a hit for a protein.

    Parameters
    ----------
    outlier_peptide_intensities : pd.DataFrame
        A dataframe with the outliers peptide intensities.
    peptide_df : pd.DataFrame
        A transformed peptide.txt dataframe with columns: ["Peptide", "Protein", "NumPeptides"].

    Returns
    -------
    protein_df
        A dataframe with the percentage of peptides that are a hit for a given protein.

    """
    protein_df = peptide_df[["Protein"]].drop_duplicates()
    # loop over each sample of the outlier peptide intensities and calculate the percentage of peptides that are a hit for a given protein
    for column_name in outlier_peptide_intensities.columns:
        hits_per_protein = pd.merge(
            peptide_df,
            outlier_peptide_intensities[column_name],
            on="Peptide",
            how="left",
        )
        hits_per_protein = hits_per_protein.groupby("Protein", as_index=False).sum()
        # number of peptide hits / number of total number of peptides for a given protein
        hits_per_protein[column_name] /= hits_per_protein["NumPeptides"]
        hits_per_protein = hits_per_protein.drop("NumPeptides", axis=1)
        protein_df = pd.merge(protein_df, hits_per_protein, on="Protein")

    return protein_df.set_index("Protein")


@df_utils.normalize(how="median_column")
@df_utils.log_scaling(log_function=np.log2, filter_outliers=True)
@df_utils.copy
def get_outlier_peptide_intensities(
    peptide_intensities: pd.DataFrame,
    max_nan_values: int = MAX_NAN_VALUES_HIT_SELECTION,
    split_above_below: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """For each sample, finds the peptides that are more than 2 standard deviations above or below the mean.

    Parameters
    ----------
    peptide_intensities : pd.DataFrame
        A dataframe containing Peptides as index and intensities as values.
    max_nan_values : int
        The maximum number of NaN values a peptide can have across samples. (Default value = MAX_NAN_VALUES_HIT_SELECTION).
    split_above_below : bool
        If True, separate between outliers below and above the mean (returns two dataframes). (Default value = False).

    Returns
    -------
    Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]
        A dataframe with the outliers peptides.

    """
    # drop peptides with more than MAX_NAN_VALUES_HIT_SELECTION NaN values
    peptide_intensities = peptide_intensities.dropna(
        thresh=peptide_intensities.shape[1] - max_nan_values, axis=0
    )

    # calculate mean and std for each peptide across samples
    peptide_mean = peptide_intensities.mean(axis=1)
    peptide_std = peptide_intensities.std(axis=1)

    # calculate lower and upper bound (2 std away from the mean)
    lower_bound = (peptide_mean - 2 * peptide_std).values.reshape(-1, 1)
    upper_bound = (peptide_mean + 2 * peptide_std).values.reshape(-1, 1)

    if split_above_below:
        peptide_intensities_above_mean = (peptide_intensities > upper_bound).astype(int)
        peptide_intensities_below_mean = (peptide_intensities < lower_bound).astype(int)
        return peptide_intensities_above_mean, peptide_intensities_below_mean
    else:
        peptide_intensities = (
            (peptide_intensities > upper_bound) | (peptide_intensities < lower_bound)
        ).astype(int)
        return peptide_intensities


def hit_selection(
    peptide_df: pd.DataFrame,
    min_peptides: int = MIN_PEPTIDES_HIT_SELECTION,
    max_nan_values: int = MAX_NAN_VALUES_HIT_SELECTION,
    split_above_below: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Hit Selection algorithm. Takes a peptide intensity dataframe, with the Peptides as index and the intensities as the values.
    Calculcates how many peptides are 2 std devs above or below the mean and reports the associated protein.

    Parameters
    ----------
    peptide_df : pd.DataFrame
        A raw peptide dataframe (peptides.txt).
    min_peptides : int
        The minimum number of peptides a protein needs to have to be to be considered. (Default value = MIN_PEPTIDES_HIT_SELECTION).
    max_nan_values : int
        The maximum number of NaN values a peptide can have across samples. (Default value = MAX_NAN_VALUES_HIT_SELECTION).
    split_above_below : bool
        If True, separate between hits below and above the mean (returns two dataframes). (Default value = False).

    Returns
    -------
    Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]
        A dataframe with the percentage of peptides that are a hit for a given protein.

    """
    peptide_intensities = peptide_df.drop(["Protein"], axis=1)
    peptide_intensities = peptide_intensities.drop_duplicates(subset="Peptide")
    peptide_intensities = peptide_intensities.set_index(["Peptide"])

    # prepare protein dataframe and peptides per protein (both filtered by each protein having at least MIN_PEPTIDES peptides)
    peptide_df = peptide_df[["Peptide", "Protein"]]
    peptide_df["NumPeptides"] = peptide_df.groupby("Protein").transform("count")
    peptide_df = peptide_df[peptide_df["NumPeptides"] >= min_peptides]

    if split_above_below:
        (
            pos_outlier_peptide_intensities,
            neg_outlier_peptide_intensities,
        ) = get_outlier_peptide_intensities(
            peptide_intensities=peptide_intensities,
            max_nan_values=max_nan_values,
            split_above_below=True,
        )

        protein_df_above_mean = get_hits_for_proteins(
            outlier_peptide_intensities=pos_outlier_peptide_intensities,
            peptide_df=peptide_df,
        )
        protein_df_below_mean = get_hits_for_proteins(
            outlier_peptide_intensities=neg_outlier_peptide_intensities,
            peptide_df=peptide_df,
        )

        return protein_df_above_mean, protein_df_below_mean
    else:
        outlier_peptide_intensities = get_outlier_peptide_intensities(
            peptide_intensities=peptide_intensities,
            max_nan_values=max_nan_values,
            split_above_below=False,
        )
        protein_df = get_hits_for_proteins(
            outlier_peptide_intensities=outlier_peptide_intensities,
            peptide_df=peptide_df,
        )

        return protein_df


def subcellular_enrichment_scores(
    proteins_with_locations: pd.DataFrame, expected_fractions_of_locations: pd.DataFrame
) -> pd.DataFrame:
    """Calculate the enrichment score for each location in the whole dataframe.

    Parameters
    ----------
    proteins_with_locations : pd.DataFrame
        A data frame containing 'Protein', 'Sample' and 'Main Location'.
    expected_fractions_of_locations : pd.DataFrame
        The expected fraction that each location should represent in a dataset.

    Returns
    -------
    enrichment_scores : pd.DataFrame
        A pandas data frame of enrichment scores.

    """
    for sample in proteins_with_locations["Sample"].unique():
        sample_df = proteins_with_locations.loc[
            proteins_with_locations["Sample"] == sample
        ]
        # Calculate the fraction that each group (each location) represents of the whole dataset
        total_proteins = sample_df["Protein"].nunique()
        sample_df = sample_df.groupby("Main location", as_index=False).apply(
            lambda location: location["Protein"].nunique() / total_proteins
        )
        sample_df.columns = ["Main location", sample]
        expected_fractions_of_locations = pd.merge(
            expected_fractions_of_locations, sample_df, on="Main location", how="left"
        )
        # Calculate the enrichment score by dividing the fraction of each location in the dataset by the expected fraction of each location
        expected_fractions_of_locations[sample] /= expected_fractions_of_locations[
            "Expected Fraction"
        ]

    expected_fractions_of_locations = expected_fractions_of_locations.drop(
        ["Expected Fraction", "# of Proteins", "Total # of Proteins"], axis=1
    )
    return expected_fractions_of_locations.set_index("Main location")
