"""src/talus_utils/algorithms.py module."""

import pandas as pd


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
