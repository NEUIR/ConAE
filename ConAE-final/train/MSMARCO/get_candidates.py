import argparse
import logging
import numpy as np
import os
from generate_beir import GenericDataLoader
from tqdm import tqdm
import pickle
import json
import faiss

logger = logging.getLogger(__name__)

doc_embeds = {}

def retrieve_qd(args, query_dict, query_embeds, docids, docindex, qrels, batch_size, outpath, prefix="outfile"):
    qids = [query[0] for query in query_dict.items()]
    qembeds = [query_embeds[qid] for qid in qids]
    doc_map_dict = {id: step for step, id in enumerate(docids)}
    query_results = {}

    for start_pos in tqdm(range(0, len(qids), batch_size)):
        batch_qembeds = qembeds[start_pos: start_pos + batch_size]
        batch_qembeds = np.array(batch_qembeds)
        batch_qids = qids[start_pos: start_pos + batch_size]
        D, I, V = docindex.search_and_reconstruct(batch_qembeds, args.topk)
        for qid, distances, indexes, vectors in zip(batch_qids, D, I, V):
            if qid not in query_results:
                query_results[qid] = {"rank_docs": [], "golden_docs": []}
            for score, idx, vector in zip(distances, indexes, vectors):
                if idx != -1:
                    did = docids[idx]
                    if did not in doc_embeds:
                        doc_embeds[did] = np.array(vector)
                    query_results[qid]["rank_docs"].append((did, float(score)))

            for did in qrels[qid].keys():
                if did not in doc_embeds:
                    doc_id = doc_map_dict[did]
                    doc_embedding = docindex.reconstruct(doc_id)
                    doc_embeds[did] = np.array(doc_embedding)
                query_results[qid]["golden_docs"].append(did)


    qresult_path = os.path.join(outpath, prefix + "_retrieval.json")
    with open(qresult_path, "w") as fout:
        for qid, results in query_results.items():
            example = {"qid": qid, "rank_docs": results["rank_docs"], "golden_docs": results["golden_docs"]}
            example = json.dumps(example)
            fout.write(example + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=str, required=True, help="Path to prebuilt index.")
    parser.add_argument('--qembed_path', type=str, required=True, help="Path to query embeddings.")
    parser.add_argument('--output_path', type=str, metavar='path', required=True, help="Path to output file.")
    parser.add_argument('--topk', type=int, required=True, help="Top k.")
    parser.add_argument('--batch_size', type=int, required=False, default=1,
                        help="Search batch of queries in parallel.")
    parser.add_argument('--threads', type=int, metavar='num', required=False, default=1,
                        help="Maximum threads to use during search.")
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
    logger.info(args)

    logger.info('Loading prebuilt index!')
    index_path = os.path.join(args.index, 'index')
    docid_path = os.path.join(args.index, 'docid')
    docindex = faiss.read_index(index_path)
    with open(docid_path, 'r') as fin:
        docids = [line.rstrip() for line in fin.readlines()]


    logger.info('Loading query embeddings!')
    with open(args.qembed_path, "rb") as fin:
        query_embedds = pickle.load(fin)

    faiss.omp_set_num_threads(args.threads)

    logger.info('Retrieving train set!')
    logger.info('Waiting a long time...')
    _, train_queries, train_qrels = GenericDataLoader(data_folder="../data/datasets/MSMARCO").load(split="train")
    retrieve_qd(args, train_queries, query_embedds, docids, docindex, train_qrels, args.batch_size, args.output_path, "train")
