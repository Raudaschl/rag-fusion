"""Download, load, and ingest the NFCorpus (BEIR) dataset for evaluation."""

import json
import os
import random
import urllib.request
import zipfile

import chromadb


def download_nfcorpus(data_dir="./datasets"):
    """Download and unzip NFCorpus from the BEIR repository."""
    dest = os.path.join(data_dir, "nfcorpus")
    if os.path.exists(dest):
        print(f"NFCorpus already exists at {dest}, skipping download.")
        return

    os.makedirs(data_dir, exist_ok=True)
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip"
    zip_path = os.path.join(data_dir, "nfcorpus.zip")

    print(f"Downloading NFCorpus from {url} ...")
    urllib.request.urlretrieve(url, zip_path)
    print("Download complete. Extracting ...")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)

    os.remove(zip_path)
    print(f"NFCorpus extracted to {dest}")


def load_nfcorpus(data_dir="./datasets"):
    """Load NFCorpus and return (corpus, queries, qrels).

    corpus:  {doc_id: text}
    queries: {query_id: text}
    qrels:   {query_id: {doc_id: relevance_score}}
    """
    base = os.path.join(data_dir, "nfcorpus")

    # --- corpus ---
    corpus = {}
    with open(os.path.join(base, "corpus.jsonl"), "r") as f:
        for line in f:
            obj = json.loads(line)
            corpus[obj["_id"]] = obj["title"] + " " + obj["text"]

    # --- queries ---
    queries = {}
    with open(os.path.join(base, "queries.jsonl"), "r") as f:
        for line in f:
            obj = json.loads(line)
            queries[obj["_id"]] = obj["text"]

    # --- qrels (test split) ---
    qrels = {}
    with open(os.path.join(base, "qrels", "test.tsv"), "r") as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            qid, did, score = parts[0], parts[1], int(parts[2])
            qrels.setdefault(qid, {})[did] = score

    print(f"Loaded NFCorpus: {len(corpus)} docs, {len(queries)} queries, {len(qrels)} qrel entries")
    return corpus, queries, qrels


def load_into_chromadb(corpus, db_path="./chroma_eval_db"):
    """Ingest corpus into a persistent ChromaDB collection and return it."""
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection("nfcorpus")

    if collection.count() > 0:
        print(f"ChromaDB collection already has {collection.count()} documents, skipping ingestion.")
        return collection

    ids = list(corpus.keys())
    docs = [corpus[did] for did in ids]
    batch_size = 500
    total = len(ids)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        collection.add(ids=ids[start:end], documents=docs[start:end])
        print(f"Ingested {end}/{total} documents into ChromaDB")

    return collection


def sample_queries(queries, qrels, n=50, seed=42):
    """Return a random sample of n query IDs present in both queries and qrels."""
    valid_ids = sorted(set(queries.keys()) & set(qrels.keys()))
    rng = random.Random(seed)
    return rng.sample(valid_ids, min(n, len(valid_ids)))
