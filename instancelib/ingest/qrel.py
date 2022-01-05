from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Sequence, Set, Tuple, TypeVar
import numpy as np

from pandas.core.frame import DataFrame
from instancelib.environment.text import TextEnvironment

from instancelib.utils.func import list_unzip3

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

    
class TrecDataset():

    def __init__(self, 
                 docids: Dict[str, Sequence[str]],
                 texts: Dict[str, Dict[str, Dict[str, str]]],
                 qrels: Dict[str, pd.DataFrame],
                 topics: pd.DataFrame,
                 pos_label: str = "Relevant",
                 neg_label: str = "Irrelevant") -> None:
        self.docids = docids
        self.texts = texts
        self.qrels = qrels
        self.topics = topics

        self.pos_label = pos_label
        self.neg_label = neg_label

        self.topic_keys = list(self.topics.id)

        self.docmap = build_doc_map(self.texts)

    def get_topicqrels(self, topic_key: str) -> str:
        return self.qrels[topic_key]

    def get_labels(self, topic_key: str, document: str) -> FrozenSet[str]:
        qrel_df = self.qrels[topic_key]
        relevancy = qrel_df.xs(document).Relevancy
        if relevancy == 1:
            return frozenset([self.pos_label])
        return frozenset([self.neg_label])

    def get_documents(self, topic_key: str) -> FrozenSet[str]:
        if topic_key in self.docids:
            return frozenset(self.docids[topic_key])
        if topic_key in self.qrels:
            return frozenset(self.qrels[topic_key].index)

    

    def get_document(self, topic_key: str, doc_id: str) -> str:
        topics = list(self.docmap[doc_id])
        if len(topics) == 1:
            doc = self.texts[topics[0]][doc_id]
        elif topic_key in topics:
            doc = self.texts[topic_key][doc_id]
        else:
            raise KeyError(f"{topic_key} not in {topics}")
        title = doc["title"]
        content = doc["content"]
        return f"{title} {content}"

    def get_env(self, topic_key: str) -> TextEnvironment[str, np.ndarray, str]:
        def yielder():
            def get_all(doc_id: str):
                data = self.get_document(topic_key, doc_id)
                labels = self.get_labels(topic_key, doc_id)
                return doc_id, data, labels
            for doc_id in self.get_documents(topic_key):
                try:
                    data_tuple = get_all(doc_id)
                except KeyError:
                    pass
                else:
                    yield data_tuple
        indices, data, labels = list_unzip3(yielder())
        env = TextEnvironment.from_data([self.neg_label, self.pos_label],
                                        indices, data, labels, None)
        return env

    def get_envs(self) -> Dict[str, TextEnvironment[str, np.ndarray, str]]:
        return {tk: self.get_env(tk) for tk in self.topic_keys}

    
    @classmethod
    def from_path(cls, base_dir: Path): 
        qrel_dir = base_dir / "qrels"
        doctexts_dir = base_dir / "doctexts"
        topics_dir = base_dir / "topics"
        docids_dir = base_dir / "docids"
        doc_ids = {f.name: read_docids(f) for f in docids_dir.iterdir()}
        texts = {f.name: read_doctexts(f) for f in doctexts_dir.iterdir()}
        qrels = {f.name: read_qrel(f) for f  in qrel_dir.iterdir()}
        topics = read_topics(topics_dir)
        return cls(doc_ids, texts, qrels, topics)