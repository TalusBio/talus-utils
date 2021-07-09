"""src/talus_utils/elib.py"""
import sqlite3
import tempfile

from sqlite3.dbapi2 import Cursor
from typing import Optional, Union

import pandas as pd

from talus_aws_utils.s3 import _read_object


class Elib:
    """A class to handle easy interactions with .elib files."""

    def __init__(self, key_or_filename: str, bucket: Optional[str] = None):
        """Initializes a new SQLite connection to a file by downloading it as a tmp file.

        Args:
            key_or_filename (str): Either a key to an object in S3 (when bucket is given)
                                   or a file name to connect to.
            bucket (str, optional): The name of the S3 bucket to load the file from. Defaults to None.
        """
        if not bucket:
            self.file_name = key_or_filename
        else:
            elib = _read_object(bucket=bucket, key=key_or_filename)
            elib_content = elib.read()
            self.tmp = tempfile.NamedTemporaryFile()
            self.tmp.write(elib_content)
            self.file_name = self.tmp.name

        # connect to tmp file
        self.connection = sqlite3.connect(self.file_name)
        self.cursor = self.connection.cursor()

    def execute_sql(
        self, sql: str, use_pandas: Optional[bool] = False
    ) -> Union[pd.DataFrame, Cursor]:
        """Executes a given SQL command and returns the result as a cursor or a pandas DataFrame.

        Args:
            sql (str): SQL String to excute.
            use_pandas (bool, optional): If True, return the query result as a pandas DataFrame.
                                         Defaults to False.

        Returns:
            Union[pd.DataFrame, Cursor]: Returns either a cursor or a pandas DataFrame with the result
                                         of the executed SQL query.
        """
        if use_pandas:
            return pd.read_sql_query(sql=sql, con=self.connection)
        else:
            return self.cursor.execute(sql)

    def close(self) -> None:
        """Closes and removes the tmp file and the connection."""
        self.tmp.close()
