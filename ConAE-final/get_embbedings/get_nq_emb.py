import os
import argparse
import numpy as np
import logging

import torch
from tqdm import tqdm
import faiss
import pickle

import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))

from model.distill_model import ConAE_model
from util.data_loader import load_nq_queries, GenericDataLoader
from util.nq_eval import validate

logger = logging.getLogger(__name__)

def encode_queries(model, query_embeds, batch_size, qids):
    qembeds =[]
    model.eval()
    for start in tqdm(range(0, len(query_embeds), batch_size)):
        batch_qids = qids[start: start + batch_size]
        batch_qembeds = query_embeds[start: start + batch_size]
        batch_qembeds = torch.tensor(np.array(batch_qembeds))
        batch_qembeds = model.enocde_queries(batch_qembeds.cuda())
        batch_qembeds = batch_qembeds.cpu().detach().numpy()
        assert len(batch_qembeds) == len(batch_qids)
        qembeds.extend(batch_qembeds)
    return qids, qembeds


def encode_dos(model, doc_embeds, batch_size, dids):
    dembeds =[]
    model.eval()
    for start in tqdm(range(0, len(doc_embeds), batch_size)):
        batch_dids = dids[start: start + batch_size]
        batch_dembeds = doc_embeds[start: start + batch_size]
        batch_dembeds = torch.tensor(np.array(batch_dembeds))
        batch_dembeds = model.enocde_docs(batch_dembeds.cuda())
        batch_dembeds = batch_dembeds.cpu().detach().numpy()
        assert len(batch_dembeds) == len(batch_dids)
        dembeds.extend(batch_dembeds)
    return dids, dembeds




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_embed_path', type=str, required=True, help='Query embedding path.')
    parser.add_argument('--doc_embed_path', type=str, required=True, help='Document embedding path.')
    parser.add_argument('--checkpoint', type=str, default=None, required=True,
                        help='Checkpoint path.')
    parser.add_argument("--batch_size", default=128, type=int, help="Total batch size for training.")
    parser.add_argument("--input_dim", default=768, type=int, help="Input dimension.")
    parser.add_argument("--output_dim", type=int, required=True, help="Output dimension.")
    parser.add_argument('--output_path', type=str, required=True, help='Output embedding path.')
    parser.add_argument("--evaluate", action="store_true", help="Get evaluation results.")
    args = parser.parse_args()

    logger.info(args)
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.DEBUG,
                        datefmt='%d-%m-%Y %H:%M:%S')

    logger.info('Loading datasets!')
    qids, answers, text = load_nq_queries('../data/datasets/NQ/nq-test.json')

    faiss.omp_set_num_threads(16)
    cpu_index = faiss.IndexFlatIP(args.output_dim)
    logger.info("Loading query embeddings...")


    query = pickle.load(open(os.path.join(args.query_embed_path, 'qembed.pkl'), 'rb'))
    qembeds = []
    for qid in qids:
        qembeds.append(query[qid])

    logger.info("Loading doc embeddings...")
    index_path = os.path.join(args.doc_embed_path, 'index')
    docid_path = os.path.join(args.doc_embed_path, 'docid')
    docindex = faiss.read_index(index_path)
    doc_embeds = docindex.reconstruct_n(0, docindex.ntotal)
    dids = []
    with open(docid_path, 'r') as fin:
        for line in fin:
            dids.append(line.strip())

    logger.info('Initializing ConAE model!')
    model = ConAE_model(args).cuda()
    model.load_state_dict(torch.load(args.checkpoint)['model'], strict=False)
    logger.info('Encoding queries...')
    qids, qembeds = encode_queries(model, qembeds, args.batch_size, qids)
    logger.info('Encoding docs...')
    dids, dembeds = encode_dos(model, doc_embeds, args.batch_size, dids)


    logger.info("Saving Queries Embeddings...")
    qid_emb = dict()
    for id, emb in zip(qids, qembeds):
        qid_emb[id] = emb
    with open(os.path.join(args.output_path, 'nq_qembed.pkl'), 'wb') as f:
        pickle.dump(qid_emb, f)

    logger.info("Saving Docs Embeddings...")
    did_emb = dict()
    for id, emb in zip(dids, dembeds):
        did_emb[id] = emb
    with open(os.path.join(args.output_path, 'nq_dembed.pkl'), 'wb') as f:
        pickle.dump(did_emb, f)

    if args.evaluate:
        del docindex
        cpu_index.add(np.array(dembeds))
        topN = 100
        _, dev_I = cpu_index.search(np.array(qembeds), topN)
        
        del qembeds, dembeds
        passage = {}
        with open('../data/datasets/NQ/psgs_w100.tsv') as fin:
            for step, line in tqdm(enumerate(fin)):
                tokens = line.strip().split("\t")
                passage[tokens[0]] = tokens[1]
        
        logger.info('Compute results...')
        result = validate(passage, answers, dev_I, qids, dids)
        logger.info('Top 5 {}'.format(result[4]))
        logger.info('Top 20 {}'.format(result[19]))
        logger.info('Top 100 {}'.format(result[99]))
