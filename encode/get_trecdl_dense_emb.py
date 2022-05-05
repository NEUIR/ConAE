import argparse
import csv
import logging
import pickle
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
from util.data_loader import GenericDataLoader
from encode.model import SentenceBERT


logger = logging.getLogger(__name__)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, metavar='path', required=True, help="Path to output file.")
    parser.add_argument('--query_path', type=str, required=True, help="Path to query file.")
    parser.add_argument('--batch_size', type=int, required=False, default=128,
                        help="Total batch size")
    parser.add_argument('--corpus_chunk_size', type=int, required=False, default=50000,
                        help="Encode corpus chunk size")
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
    logger.info(args)

    logger.info("Loading Datasets...")
    queries = dict()
    with open(args.query_path, "r") as f:
        query = csv.reader(f, delimiter='\t')
        for i in query:
            queries[i[0]] = i[1]


    logger.info("Loading ANCE Model...")
    model_path = "msmarco-roberta-base-ance-firstp"
    model = SentenceBERT(model_path)

    logger.info("Encoding Queries...")
    query_ids = list(queries.keys())
    results = {qid: {} for qid in query_ids}
    queries = [queries[qid] for qid in queries]
    query_embeddings = model.encode_queries(queries, batch_size=args.batch_size,show_progress_bar=True,convert_to_tensor=False)

    logger.info("Saving Queries Embeddings...")
    qid_emb = dict()
    for id,emb in zip(query_ids,query_embeddings):
        qid_emb[id]=emb
    with open(os.path.join(args.output_path, 'qembed.pkl'), 'wb') as f:
        pickle.dump(qid_emb, f)
