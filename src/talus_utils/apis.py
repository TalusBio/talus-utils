"""src/talus_utils/apis.py module."""
from typing import Dict, List, Optional

import requests


def query_panther(
    input_list: List[str],
    organism: str = "9606",
    annot_type: str = "cellular_component",
    go_filters: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Query panther for GO analysis.
    API specs: http://pantherdb.org/services/openAPISpec.jsp.

    Parameters
    ----------
    input_list : List[str]
        A list of Protein Accessions or Gene ID's according to the Panther API specs.
    organism : str, optional
        The Identifier of the target organism, by default "9606"
    annot_type : str, optional
        The name of the annotation type to query for, by default "cellular_component".
        Should be one of ['molecular_function', 'biological_process', 'cellular_component'].
    go_filters : Optional[List[str]], optional
        A list of filters to apply to the results, e.g. ['nucleus', 'lysosome'], by default None

    Returns
    -------
    Dict[str, str]
        A resulting dict with the enrichment scores for the input list.
    """
    # Get annotation dataset based on the given annotation type
    supported_dataset_url = (
        "http://pantherdb.org/services/oai/pantherdb/supportedannotdatasets"
    )
    response = requests.get(supported_dataset_url)
    annot_data_set = [
        obj["id"]
        for obj in response.json()["search"]["annotation_data_sets"][
            "annotation_data_type"
        ]
        if obj["label"] == annot_type
    ][0]

    url = "http://pantherdb.org/services/oai/pantherdb/enrich/overrep?geneInputList={}&organism={}&annotDataSet={}"
    response = requests.post(url.format(input_list, organism, annot_data_set))
    results = response.json()["results"]["result"]
    if go_filters:
        results = [obj for obj in results if obj["term"]["label"] in go_filters]

    return results
