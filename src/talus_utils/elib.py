"""src/talus_utils/elib.py module."""
import sqlite3
import tempfile

from pathlib import Path
from sqlite3.dbapi2 import Cursor
from typing import Dict, Optional, Union

import pandas as pd

from talus_utils.s3 import _read_object


class Elib:
    """Handle easy interactions with .elib files."""

    def __init__(self, key_or_filename: Union[Path, str], bucket: Optional[str] = None):
        """Initialize a new SQLite connection to a file by downloading it as a tmp file.

        Parameters
        ----------
        key_or_filename : str
            Either a key to an object in S3 (when bucket is given) or a file name to connect to.
        bucket : Optional[str], optional
            The name of the S3 bucket to load the file from, by default None
        """
        self._tmp = None
        if not bucket:
            self._file_name = key_or_filename
        else:
            elib = _read_object(bucket=bucket, key=key_or_filename)
            elib_content = elib.read()
            self._tmp = tempfile.NamedTemporaryFile()
            self._tmp.write(elib_content)
            self._file_name = self._tmp.name

        # connect to tmp file
        self._connection = sqlite3.connect(self._file_name)
        self._cursor = self._connection.cursor()

    def execute_sql(
        self, sql: str, use_pandas: Optional[bool] = False
    ) -> Union[pd.DataFrame, Cursor]:
        """Execute a given SQL command and returns the result as a cursor or a pandas DataFrame.

        Parameters
        ----------
        sql : str
            SQL String to excute.
        use_pandas : bool
            If True, return the query result as a pandas DataFrame. (Default value = False).

        Returns
        -------
        Union[pd.DataFrame, Cursor]
            Returns either a cursor or a pandas DataFrame with the result
            of the executed SQL query.

        """
        if use_pandas:
            return pd.read_sql_query(sql=sql, con=self._connection)
        else:
            return self._cursor.execute(sql)

    def close(self) -> None:
        """Close and remove the tmp file and the connection."""
        if self._tmp:
            self._tmp.close()


def get_unique_peptide_proteins(
    elib_filename: Union[Path, str], bucket: Optional[str] = None
) -> Dict[str, Union[int, str]]:
    """Get the number of unique peptides and proteins in the given elib file.

    Parameters
    ----------
    elib_filename : Union[Path, str]
        The path to the elib file.
    bucket : Optional[str], optional
        The name of the bucket to use. (Default value = None)

    Returns
    -------
    Dict[str, Union[int, str]]
        A dictionary containing the sample name, number of unique peptides and proteins.
    """
    elib_conn = Elib(key_or_filename=elib_filename, bucket=bucket)
    peptide_to_protein = elib_conn.execute_sql(
        sql="SELECT PeptideSeq, ProteinAccession FROM peptidetoprotein WHERE isDecoy == 0;",
        use_pandas=True,
    )
    sample_name = Path(elib_filename).with_suffix("").stem
    unique_proteins = peptide_to_protein["ProteinAccession"].nunique()
    unique_peptides = peptide_to_protein["PeptideSeq"].nunique()
    elib_conn.close()
    return {
        "Sample Name": sample_name,
        "Unique Proteins": unique_proteins,
        "Unique Peptides": unique_peptides,
    }
