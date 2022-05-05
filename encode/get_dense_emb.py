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
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name like MSMARCO, TREC_COVID.")
    parser.add_argument('--output_path', type=str, metavar='path', required=True, help="Path to output file.")
    parser.add_argument('--batch_size', type=int, required=False, default=128, help="Total batch size")
    parser.add_argument('--corpus_chunk_size', type=int, required=False, default=50000,
                        help="Encode corpus chunk size")
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
    logger.info(args)

    logger.info("Loading Datasets...")
    if args.dataset.lower() == 'msmarco':
        corpus, queries, qrels = GenericDataLoader(data_folder="../data/datasets/msmarco").load(split="dev")
    elif args.dataset.lower() == 'trec_covid':
        corpus, queries, qrels = GenericDataLoader(data_folder="../data/datasets/trec_covid").load(split="test")

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

    logger.info("Sorting Corpus by document length (Longest first)...")
    corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), reverse=True)
    corpus = [corpus[cid] for cid in corpus_ids]

    logger.info("Encoding Corpus in batches... Warning: This might take a while!")
    itr = range(0, len(corpus), args.corpus_chunk_size)

    for batch_num, corpus_start_idx in enumerate(itr):
        logger.info("Encoding Batch {}/{}...".format(batch_num + 1, len(itr)))
        corpus_end_idx = min(corpus_start_idx + args.corpus_chunk_size, len(corpus))

        # Encode chunk of corpus
        sub_corpus_embeddings = model.encode_corpus(
            corpus[corpus_start_idx:corpus_end_idx],
            batch_size=args.batch_size,
            show_progress_bar=True,
            convert_to_tensor=False
        )

        if not batch_num:
           corpus_embeddings = sub_corpus_embeddings
        else:
           corpus_embeddings = np.vstack([corpus_embeddings, sub_corpus_embeddings])

    logger.info("Saving Docs Embeddings...")
    did_emb = dict()
    for id, emb in zip(corpus_ids, corpus_embeddings):
        did_emb[id] = emb
    with open(os.path.join(args.output_path, 'dembed.pkl'), 'wb') as f:
        pickle.dump(did_emb, f)
