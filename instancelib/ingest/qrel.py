from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, TypeVar

from pandas.core.frame import DataFrame

from ..environment import AbstractEnvironment
from ..instances import Instance
import json
from ..typehints.typevars import VT, LT

import pandas as pd

IT = TypeVar("IT", bound="Instance[Any, Any, Any, Any]", covariant=True)

class TrecDataset():

    def __init__(self, 
                 docids: Dict[str, Sequence[str]],
                 texts: Dict[str, pd.DataFrame],
                 qrels: Dict[str, pd.DataFrame],
                 topics: Dict[str, Dict[str, ]]) -> None:
        pass

def read_doctexts(doctext_file: Path) -> Dict[str, pd.DataFrame]:
    try:
        with doctext_file.open() as f:
            jsons = [json.loads(line) for line in f.readlines()]
        df = pd.DataFrame(jsons)
    except Exception as e:
        print(e)
        return pd.DataFrame()
    return df

def read_docids(docid_file: Path) -> Sequence[str]:
    with docid_file.open() as f:
        docids = [line.replace("\n", "") for line in f.readlines()]
    return docids

def read_qrel(qrel_file: Path) -> pd.DataFrame:
    return pd.read_csv(qrel_file, sep="\t")

def read_topics(topic_dir: Path) -> pd.DataFrame:
    jsons = list()
    for file in topic_dir.iterdir():
        with file.open() as f:
            jsons.append(*[json.loads(line) for line in f.readlines()])
    return pd.DataFrame(jsons)

def read_qrel_dataset(base_dir: Path): 
    qrel_dir = base_dir / "qrels"
    doctexts_dir = base_dir / "doctexts"
    topics_dir = base_dir / "topics"
    docids_dir = base_dir / "docids"
    doc_ids = {f.name: read_docids(f) for f in docids_dir.iterdir()}
    texts = {f.name: read_doctexts(f) for f in doctexts_dir.iterdir()}
    qrels = {f.name: read_qrel(f) for f  in qrel_dir.iterdir()}
    topics = read_topics(topics_dir)
    return doc_ids, texts, qrels, topics

    
