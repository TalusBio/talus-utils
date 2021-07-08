"""src/talus_utils/uniprot.py"""
from typing import Tuple


def parse_fasta_header(fasta_header: str) -> Tuple[str, str, str]:
    """Parse a fasta header with the following format: db|UniqueIdentifier|EntryName.
    https://www.uniprot.org/help/fasta-headers

    Args:
        fasta_header (str): A fasta header in the format: db|UniqueIdentifier|EntryName

    Raises:
        ValueError: If the fasta header doesn't follow the format: db|UniqueIdentifier|EntryName.

    Returns:
        Tuple[str, str, str]: The db, unique identifier and entry name of the fasta header.
    """
    db, unique_identifier, entry_name = fasta_header.split("|")
    if not db or not unique_identifier or not entry_name:
        raise ValueError(
            "Invalid Fasta Header. It needs to follow the format: db|UniqueIdentifier|EntryName."
        )
    return db, unique_identifier, entry_name


def parse_fasta_header_uniprot_entry(fasta_header: str) -> Tuple[str, str]:
    """Extracts the Protein and Species name from a fasta header in the format:
    db|UniqueIdentifier|EntryName. The EntryName field has the format
    ProteinName_SpeciesName according to https://www.uniprot.org/help/entry_name
    and this function extracts the Protein and Species name.

    Args:
        fasta_header (str): [description]

    Raises:
        ValueError: If the fasta header EntryName doesn't follow the format: ProteinName_SpeciesName.

    Returns:
        Tuple[str, str]: The ProteinName and SpeciesName of the EntryName field
        of the given fasta header.
    """
    _, _, entry_name = parse_fasta_header(fasta_header=fasta_header)
    protein_name, species_name = entry_name.split("_")
    if not protein_name or not species_name:
        raise ValueError(
            "Invalid Fasta Header Entry Name. It needs to follow the format: ProteinName_SpeciesName."
        )
    return protein_name, species_name
