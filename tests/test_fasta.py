"""Test cases for the fasta module."""
import pytest

from talus_utils.fasta import parse_fasta_header, parse_fasta_header_uniprot_entry


def test_parse_fasta_header_valid() -> None:
    """Tests parse_fasta_header."""
    fasta_header = "sp|A0A096LP01|SIM26_HUMAN"
    expected_db = "sp"
    expected_unique_identifier = "A0A096LP01"
    expected_entry_name = "SIM26_HUMAN"
    actual_db, actual_unique_identifier, actual_entry_name = parse_fasta_header(
        fasta_header=fasta_header
    )
    assert expected_db == actual_db
    assert expected_unique_identifier == actual_unique_identifier
    assert expected_entry_name == actual_entry_name


def test_parse_fasta_header_invalid_db() -> None:
    """Tests parse_fasta_header with invalid db field."""
    expected_error = r"not enough values to unpack \(expected 3, got 2\)"
    invalid_fasta_header_nodb = "A0A096LP01|SIM26_HUMAN"
    with pytest.raises(ValueError, match=expected_error):
        _ = parse_fasta_header(fasta_header=invalid_fasta_header_nodb)

    expected_error = r"Invalid Fasta Header. It needs to follow the format: db\|UniqueIdentifier\|EntryName."
    invalid_fasta_header_emptydb = "|A0A096LP01|SIM26_HUMAN"
    with pytest.raises(ValueError, match=expected_error):
        _ = parse_fasta_header(fasta_header=invalid_fasta_header_emptydb)


def test_parse_fasta_header_invalid_id() -> None:
    """Tests parse_fasta_header with invalid id field."""
    expected_error = r"not enough values to unpack \(expected 3, got 2\)"
    invalid_fasta_header_noid = "sp|SIM26_HUMAN"
    with pytest.raises(ValueError, match=expected_error):
        _ = parse_fasta_header(fasta_header=invalid_fasta_header_noid)

    expected_error = r"Invalid Fasta Header. It needs to follow the format: db\|UniqueIdentifier\|EntryName."
    invalid_fasta_header_emptyid = "sp||SIM26_HUMAN"
    with pytest.raises(ValueError, match=expected_error):
        _ = parse_fasta_header(fasta_header=invalid_fasta_header_emptyid)


def test_parse_fasta_header_invalid_entry() -> None:
    """Tests parse_fasta_header with invalid entry field."""
    expected_error = r"not enough values to unpack \(expected 3, got 2\)"
    invalid_fasta_header_noentry = "sp|A0A096LP01"
    with pytest.raises(ValueError, match=expected_error):
        _ = parse_fasta_header(fasta_header=invalid_fasta_header_noentry)

    expected_error = r"Invalid Fasta Header. It needs to follow the format: db\|UniqueIdentifier\|EntryName."
    invalid_fasta_header_emptyentry = "sp|A0A096LP01|"
    with pytest.raises(ValueError, match=expected_error):
        _ = parse_fasta_header(fasta_header=invalid_fasta_header_emptyentry)


def test_parse_fasta_header_uniprot_entry_valid() -> None:
    """Tests parse_fasta_header_uniprot_entry."""
    fasta_header = "sp|A0A096LP01|SIM26_HUMAN"
    expected_protein = "SIM26"
    expected_species = "HUMAN"
    actual_protein, actual_species = parse_fasta_header_uniprot_entry(
        fasta_header=fasta_header
    )
    assert expected_protein == actual_protein
    assert expected_species == actual_species


def test_parse_fasta_header_uniprot_entry_invalid_protein() -> None:
    """Tests parse_fasta_header_uniprot_entry with invalid protein field."""
    expected_error = r"not enough values to unpack \(expected 2, got 1\)"
    invalid_fasta_header_noprotein = "sp|A0A096LP01|HUMAN"
    with pytest.raises(ValueError, match=expected_error):
        _ = parse_fasta_header_uniprot_entry(
            fasta_header=invalid_fasta_header_noprotein
        )

    expected_error = r"Invalid Fasta Header Entry Name. It needs to follow the format: ProteinName_SpeciesName."
    invalid_fasta_header_emptyprotein = "sp|A0A096LP01|_HUMAN"
    with pytest.raises(ValueError, match=expected_error):
        _ = parse_fasta_header_uniprot_entry(
            fasta_header=invalid_fasta_header_emptyprotein
        )


def test_parse_fasta_header_uniprot_entry_invalid_species() -> None:
    """Tests parse_fasta_header_uniprot_entry with invalid species field."""
    expected_error = r"not enough values to unpack \(expected 2, got 1\)"
    invalid_fasta_header_nospecies = "sp|A0A096LP01|SIM26"
    with pytest.raises(ValueError, match=expected_error):
        _ = parse_fasta_header_uniprot_entry(
            fasta_header=invalid_fasta_header_nospecies
        )

    expected_error = r"Invalid Fasta Header Entry Name. It needs to follow the format: ProteinName_SpeciesName."
    invalid_fasta_header_emptyspecies = "sp|A0A096LP01|SIM26_"
    with pytest.raises(ValueError, match=expected_error):
        _ = parse_fasta_header_uniprot_entry(
            fasta_header=invalid_fasta_header_emptyspecies
        )
