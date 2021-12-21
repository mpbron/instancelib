from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple, TypeVar

from pandas.core.frame import DataFrame

from ..environment import AbstractEnvironment
from ..instances import Instance
import json
from ..typehints.typevars import VT, LT

import pandas as pd

IT = TypeVar("IT", bound="Instance[Any, Any, Any, Any]", covariant=True)

def read_doctexts(doctext_file: Path) -> Dict[str, Dict[str, str]]:
    def process_line(line: str) -> Tuple[str, Dict[str, str]]:
        obj = json.loads(line)
        return obj["id"], obj
    with doctext_file.open() as f:
        tuples = [process_line(line) for line in f.readlines()]
    dictionary = {key: obj for (key, obj) in tuples}
    return dictionary

def build_doc_map(topic_docs: Dict[str, Dict[str, Dict[str, str]]]) -> Dict[str, Set[str]]:
    docmap : Dict[str, Set[str]] = dict()
    for topic, docs_dict in topic_docs.items():
        for doc_key in docs_dict:
            docmap.setdefault(doc_key, set()).add(topic)
    return docmap

def read_docids(docid_file: Path) -> Sequence[str]:
    with docid_file.open() as f:
        docids = frozenset([line.replace("\n", "") for line in f.readlines()])
    return docids

def read_qrel(qrel_file: Path) -> pd.DataFrame:
    df = pd.read_csv(qrel_file, 
                       sep="\t", 
                       header=None,
                       names=["Topic", "Iteration", "Document", "Relevancy"],
                       dtype={"Topic": "str", "Document": "str"})
    df = df.set_index("Document")
    return df

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

    
