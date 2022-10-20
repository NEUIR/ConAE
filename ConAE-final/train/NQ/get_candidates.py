import argparse
import json
import logging
import os
import pickle

import faiss
import numpy as np
from tqdm import tqdm

from util.dpr_utils import has_answer, SimpleTokenizer

logger = logging.getLogger(__name__)


def load_nq_queries(path):
    qids = []
    answers = {}
    with open(path) as fin:
        for line in fin:
            example = json.loads(line.strip())
            qids.append(example["qid"])
            answers[example["qid"]] = example["answers"]
    return qids, answers


def retrieve_candidates(args, inpath, dids):
    query_results = {}
    qids, answers = load_nq_queries(inpath)
    qembeds = []
    for qid in qids:
        qembeds.append(query_embeds[qid])

    args.topk = 500
    D, I = docindex.search(np.array(qembeds), args.topk)
    for qid, distances, indexes in tqdm(zip(qids, D, I)):
        query_results[qid] = {"qid": qid, "rank_docs": [], "golden_docs": [], "answers": answers[qid]}
        for score, idx in zip(distances, indexes):
            if idx != -1:
                did = dids[idx]
                query_results[qid]["rank_docs"].append((did, float(score)))
    return query_results


def add_golden_docs(query_results, outpath):
    with open(outpath, "w") as fout:
        for qid in query_results.keys():
            for doc in query_results[qid]["rank_docs"]:
                text = passage[doc[0]]
                if has_answer(query_results[qid]["answers"], text, tokenizer):
                    query_results[qid]["golden_docs"].append(doc[0])
            example = query_results[qid]
            fout.write(json.dumps(example) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--qembed_path', type=str, required=True, help="Path to query embeddings.")
    parser.add_argument('--index', type=str, required=True, help="Path to prebuilt index.")
    parser.add_argument('--topk', type=int, required=True, help="Top k.")
    parser.add_argument('--threads', type=int, metavar='num', required=False, default=1,
                        help="Maximum threads to use during search.")
    parser.add_argument('--data_path', type=str, required=True, help="Path to datasets.")
    parser.add_argument('--output_path', type=str, metavar='path', required=True, help="Path to output file.")
    args = parser.parse_args()

    handlers = [logging.StreamHandler()]
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.DEBUG,
                        datefmt='%d-%m-%Y %H:%M:%S', handlers=handlers)

    logger.info(args)
    faiss.omp_set_num_threads(16)

    logger.info('Loading prebuilt index!')
    index_path = os.path.join(args.index, 'index')
    docid_path = os.path.join(args.index, 'docid')
    docindex = faiss.read_index(index_path)
    dids = []
    with open(docid_path, 'r') as fin:
        for line in fin:
            dids.append(line.strip())

    logger.info('Loading query embeddings!')
    with open(args.qembed_path, "rb") as fin:
        query_embeds = pickle.load(fin)

    faiss.omp_set_num_threads(args.threads)

    tok_opts = {}
    tokenizer = SimpleTokenizer(**tok_opts)

    logger.info('Retrieving train set!')
    train_path = os.path.join(args.data_path, "nq-train.json")
    train_results = retrieve_candidates(args, train_path, dids)

    logger.info('Retrieving dev set!')
    dev_path = os.path.join(args.data_path, "nq-dev.json")
    dev_results = retrieve_candidates(args, dev_path, dids)
    del docindex, query_embeds

    passage = {}
    with open("../data/datasets/NQ/psgs_w100.tsv") as fin:
        for step, line in tqdm(enumerate(fin)):
            tokens = line.strip().split("\t")
            passage[tokens[0]] = tokens[1]

    logger.info("Saving files!")
    add_golden_docs(dev_results, os.path.join(args.output_path, "./dev_retrieval.json"))
    add_golden_docs(train_results, os.path.join(args.output_path, "./train_retrieval.json"))
