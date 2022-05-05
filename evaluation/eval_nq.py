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

from model.distill_model import ConAE_model, CE_model
from util.data_loader import load_nq_queries
from util.nq_eval import validate

logger = logging.getLogger(__name__)

def encode_queries(model, query_embeds, batch_size,qids):
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



def encode_queries_pca(pca_model, query_embeds, batch_size,qids):
    qembeds = []
    for start in tqdm(range(0, len(query_embeds), batch_size)):
        batch_qids = qids[start: start + batch_size]
        batch_qembeds = query_embeds[start: start + batch_size]
        batch_qembeds = pca_model.apply_py(np.array(batch_qembeds))
        assert len(batch_qembeds) == len(batch_qids)
        qembeds.extend(batch_qembeds)
    return qids, qembeds


def encode_dos_pca(pca_model, doc_embeds, batch_size, dids):
    dembeds = []
    for start in tqdm(range(0, len(doc_embeds), batch_size)):
        batch_dids = dids[start: start + batch_size]
        batch_dembeds = doc_embeds[start: start + batch_size]
        batch_dembeds = pca_model.apply_py(np.array(batch_dembeds))
        assert len(batch_dembeds) == len(batch_dids)
        dembeds.extend(batch_dembeds)
    return dids, dembeds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_embed_path', type=str, required=True, help='Query embedding path.')
    parser.add_argument('--doc_embed_path', type=str, required=True, help='Document embedding path.')
    parser.add_argument('--nq_test_path', type=str, required=True, help='NQ test dataset queries and answers path.')
    parser.add_argument('--psgs_w100_path', type=str, required=True, help='Psgs_w100 path.')
    parser.add_argument('--checkpoint', type=str, default=None, required=True,
                        help='Checkpoint path or PCA model path.')
    parser.add_argument('--model', type=str, required=True, help='Model name like ConAE, CE, PCA.')
    parser.add_argument("--batch_size", default=128, type=int, help="Total batch size for training.")
    parser.add_argument("--input_dim", default=768, type=int, help="Input dimension.")
    parser.add_argument("--output_dim", type=int, required=True, help="Output dimension.")
    args = parser.parse_args()

    logger.info(args)
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.DEBUG,
                        datefmt='%d-%m-%Y %H:%M:%S')

    logger.info('Loading datasets!')
    qids, answers, text = load_nq_queries(args.nq_test_path)

    faiss.omp_set_num_threads(16)
    cpu_index = faiss.IndexFlatIP(args.output_dim)
    logger.info("Loading query embeddings...")

    query = pickle.load(open(os.path.join(args.query_embed_path, 'embedding.pkl'), 'rb'))

    query_embedding = query['embedding'].to_list()
    query_text = query['text'].to_list()
    qembeds = []
    for t in text:
        i = query_text.index(t)
        qembeds.append(query_embedding[i])



    logger.info("Loading doc embeddings...")
    index_path = os.path.join(args.doc_embed_path, 'index')
    docid_path = os.path.join(args.doc_embed_path, 'docid')
    docindex = faiss.read_index(index_path)
    doc_embeds = docindex.reconstruct_n(0, docindex.ntotal)
    dids = []
    with open(docid_path, 'r') as fin:
        for line in fin:
            dids.append(line.strip())

    if args.model.lower() == 'conae' or args.model.lower() == 'kl':
        logger.info('Initializing ConAE model!')
        model = ConAE_model(args).cuda()
        model.load_state_dict(torch.load(args.checkpoint)['model'], strict=False)
        logger.info('Encoding queries...')
        qids, qembeds = encode_queries(model, qembeds, args.batch_size, qids)
        logger.info('Encoding docs...')
        dids, dembeds = encode_dos(model, doc_embeds, args.batch_size, dids)
    elif args.model.lower() == 'ce':
        logger.info('Initializing CE model!')
        model = CE_model(args).cuda()
        model.load_state_dict(torch.load(args.checkpoint)['model'], strict=False)
        logger.info('Encoding queries...')
        qids, qembeds = encode_queries(model, qembeds, args.batch_size, qids)
        logger.info('Encoding docs...')
        dids, dembeds = encode_dos(model, doc_embeds, args.batch_size, dids)
    elif args.model.lower() == 'pca':
        logger.info('Initializing PCA model')
        pca_model = faiss.read_VectorTransform(args.checkpoint)
        logger.info('Encoding queries...')
        qids, qembeds = encode_queries_pca(pca_model, qembeds, args.batch_size, qids)
        logger.info('Encoding docs...')
        dids, dembeds = encode_dos_pca(pca_model, doc_embeds, args.batch_size, dids)
    else:
        qembeds = qembeds
        dembeds = doc_embeds

    # logger.info("Saving Queries Embeddings...")
    # qid_emb = dict()
    # for id, emb in zip(qids, qembeds):
    #     qid_emb[id] = emb
    # with open(os.path.join(args.output_path, 'nq_qembed.pkl'), 'wb') as f:
    #     pickle.dump(qid_emb, f)
    #
    # logger.info("Saving Docs Embeddings...")
    # did_emb = dict()
    # for id, emb in zip(dids, dembeds):
    #     did_emb[id] = emb
    # with open(os.path.join(args.output_path, 'nq_dembed.pkl'), 'wb') as f:
    #     pickle.dump(did_emb, f)

    del docindex
    cpu_index.add(np.array(dembeds))
    topN = 100
    _, dev_I = cpu_index.search(np.array(qembeds), topN)

    del qembeds,dembeds
    passage = {}
    with open(args.psgs_w100_path) as fin:
        for step, line in tqdm(enumerate(fin)):
            tokens = line.strip().split("\t")
            passage[tokens[0]] = tokens[1]

    logger.info('Compute results...')
    result = validate(passage, answers, dev_I, qids, dids)
    logger.info('Top 5 {}'.format(result[4]))
    logger.info('Top 20 {}'.format(result[19]))
    logger.info('Top 100 {}'.format(result[99]))