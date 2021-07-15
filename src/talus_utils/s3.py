"""src/talus_utils/s3.py module."""
from io import BytesIO

import boto3

from botocore.exceptions import ClientError


def _read_object(bucket: str, key: str) -> BytesIO:
    """Read an object in byte format from a given s3 bucket and key name.

    Parameters
    ----------
    bucket : str
        The S3 bucket to load from.
    key : str
        The object key within the s3 bucket.

    Returns
    -------
    BytesIO
        The object in byte format.

    Raises
    ------
    ValueError
        If the file couldn't be found.

    """
    s3_resource = boto3.Session().resource("s3")
    s3_bucket = s3_resource.Bucket(bucket)
    data = BytesIO()
    try:
        s3_bucket.download_fileobj(Key=key, Fileobj=data)
        data.seek(0)
        return data
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            raise ValueError("File doesn't exist.")
        else:
            raise
