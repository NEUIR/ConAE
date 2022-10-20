import argparse
import logging
import os
import pickle
import sys
import time

import faiss
import numpy as np
import pytrec_eval
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(sys.path[0]))
from model.distill_model import ConAE_model
from util.msmarco_eval import compute_metrics
from util.data_loader import GenericDataLoader

logger = logging.getLogger(__name__)


def EvalDevQuery(query_embedding2id, passage_embedding2id, dev_query_positive_id, I_nearest_neighbor, topN):
    prediction = {}  # [qid][docid] = docscore, here we use -rank as score, so the higher the rank (1 > 2), the higher the score (-1 > -2)

    total = 0
    labeled = 0
    Atotal = 0
    Alabeled = 0
    qids_to_ranked_candidate_passages = {}
    for query_idx in range(len(I_nearest_neighbor)):
        seen_pid = set()
        query_id = query_embedding2id[query_idx]
        prediction[query_id] = {}

        top_ann_pid = I_nearest_neighbor[query_idx].copy()
        selected_ann_idx = top_ann_pid[:topN]
        rank = 0

        if query_id in qids_to_ranked_candidate_passages:
            pass
        else:
            # By default, all PIDs in the list of 1000 are 0. Only override those that are given
            tmp = [0] * 1000
            qids_to_ranked_candidate_passages[query_id] = tmp

        for idx in selected_ann_idx:
            pred_pid = passage_embedding2id[idx]

            if not pred_pid in seen_pid:
                # this check handles multiple vector per document
                qids_to_ranked_candidate_passages[query_id][rank] = pred_pid
                Atotal += 1
                if pred_pid not in dev_query_positive_id[query_id]:
                    Alabeled += 1
                if rank < 10:
                    total += 1
                    if pred_pid not in dev_query_positive_id[query_id]:
                        labeled += 1
                rank += 1
                prediction[query_id][pred_pid] = -rank
                seen_pid.add(pred_pid)

    # use out of the box evaluation script
    evaluator = pytrec_eval.RelevanceEvaluator(
        dev_query_positive_id, {'map_cut', 'ndcg_cut', 'recip_rank', 'recall'})

    eval_query_cnt = 0
    result = evaluator.evaluate(prediction)

    qids_to_relevant_passageids = {}
    for qid in dev_query_positive_id:
        if qid in qids_to_relevant_passageids:
            pass
        else:
            qids_to_relevant_passageids[qid] = []
            for pid in dev_query_positive_id[qid]:
                qids_to_relevant_passageids[qid].append(pid)

    ms_mrr = compute_metrics(qids_to_relevant_passageids, qids_to_ranked_candidate_passages)

    ndcg = 0
    Map = 0
    mrr = 0
    recall = 0

    for k in result.keys():
        eval_query_cnt += 1
        ndcg += result[k]["ndcg_cut_10"]
        Map += result[k]["map_cut_10"]
        mrr += result[k]["recip_rank"]
        recall += result[k]["recall_" + str(topN)]

    final_ndcg = ndcg / eval_query_cnt
    final_Map = Map / eval_query_cnt
    final_mrr = mrr / eval_query_cnt
    final_recall = recall / eval_query_cnt
    hole_rate = labeled / total
    Ahole_rate = Alabeled / Atotal

    return final_ndcg, eval_query_cnt, final_Map, final_mrr, final_recall, hole_rate, ms_mrr, Ahole_rate, result, prediction


def encode_queries(model, query_embeds, batch_size, dev_queries):
    qembeds = []
    qembeds_raw = []
    qids = []
    for qid, embedding in query_embeds.items():
        if qid in dev_queries:
            qids.append(qid)
            qembeds_raw.append(embedding)
    assert len(qids) == len(qembeds_raw)
    del query_embeds
    model.eval()
    for start in tqdm(range(0, len(qembeds_raw), batch_size)):
        batch_qids = qids[start: start + batch_size]
        batch_qembeds = qembeds_raw[start: start + batch_size]
        batch_qembeds = torch.tensor(np.array(batch_qembeds))
        batch_qembeds = model.enocde_queries(batch_qembeds.cuda())
        batch_qembeds = batch_qembeds.cpu().detach().numpy()
        assert len(batch_qembeds) == len(batch_qids)
        qembeds.extend(batch_qembeds)
    return qids, qembeds


def encode_dos(model, doc_embeds, batch_size):
    dembeds = []
    dids = []
    doc_embeds = list(doc_embeds.items())
    model.eval()
    for start in tqdm(range(0, len(doc_embeds), batch_size)):
        batch_dids = []
        batch_dembeds = []
        for example in doc_embeds[start: start + batch_size]:
            batch_dids.append(example[0])
            batch_dembeds.append(example[1])
        batch_dembeds = torch.tensor(np.array(batch_dembeds))
        batch_dembeds = model.enocde_docs(batch_dembeds.cuda())
        batch_dembeds = batch_dembeds.cpu().detach().numpy()
        assert len(batch_dembeds) == len(batch_dids)
        dembeds.extend(batch_dembeds)
        dids.extend(batch_dids)
    return dids, dembeds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", action="store_true", help="Get evaluation results.")
    parser.add_argument('--query_embed_path', type=str, required=True, help='Query embedding path.')
    parser.add_argument('--doc_embed_path', type=str, required=True, help='Document embedding path.')
    parser.add_argument('--output_path', type=str, required=True, help='Compressed embedding path.')
    parser.add_argument('--checkpoint', type=str, default=None, required=True, help='Checkpoint path.')
    parser.add_argument("--batch_size", default=128, type=int, help="Total batch size for training.")
    parser.add_argument("--input_dim", default=768, type=int, help="Input dimension.")
    parser.add_argument("--output_dim", type=int, required=True, help="Output dimension.")
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
    logger.info(args)

    logger.info('Loading datasets!')
    _, dev_queries, dev_qrels = GenericDataLoader(data_folder="../data/datasets/MSMARCO").load(split="dev")

    faiss.omp_set_num_threads(16)
    cpu_index = faiss.IndexFlatIP(args.output_dim)
    logger.info("Loading query embeddings...")
    with open(args.query_embed_path, "rb") as fin:
        query_embeds = pickle.load(fin)

    logger.info("Loading doc embeddings...")
    with open(args.doc_embed_path, "rb") as fin:
        doc_embeds = pickle.load(fin)

    logger.info('Initializing ConAE model!')
    model = ConAE_model(args).cuda()
    model.load_state_dict(torch.load(args.checkpoint)['model'], strict=False)
    logger.info('Encoding queries...')
    qids, qembeds = encode_queries(model, query_embeds, args.batch_size, dev_queries)
    logger.info('Encoding docs...')
    dids, dembeds = encode_dos(model, doc_embeds, args.batch_size)

    logger.info("Saving Queries Embeddings...")
    qid_emb = dict()
    for id, emb in zip(qids, qembeds):
        qid_emb[id] = emb
    with open(os.path.join(args.output_path, 'conae_qembed.pkl'), 'wb') as f:
        pickle.dump(qid_emb, f)

    logger.info("Saving Docs Embeddings...")
    did_emb = dict()
    for id, emb in zip(dids, dembeds):
        did_emb[id] = emb
    with open(os.path.join(args.output_path, 'conae_dembed.pkl'), 'wb') as f:
        pickle.dump(did_emb, f)

    if args.evaluate:
        topN = 1000
        cpu_index.add(np.array(dembeds))
        logger.info('Testing latency...')
        start = time.perf_counter()
        _, dev_I = cpu_index.search(np.array(qembeds), topN)
        end = time.perf_counter() - start
        print('{:.6f}s for the total'.format(end))
        print('{:.6f}s for the every'.format(end / np.array(qembeds).shape[0]))

        logger.info('Compute results...')
        result = EvalDevQuery(qids, dids, dev_qrels, dev_I, topN)
        final_ndcg, eval_query_cnt, final_Map, final_mrr, final_recall, hole_rate, ms_mrr, Ahole_rate, metrics, prediction = result
        print("Reranking NDCG@10:" + str(final_ndcg))
        print("Reranking map@10:" + str(final_Map))
        print("Reranking pytrec_mrr:" + str(final_mrr))
        print("Reranking recall@" + str(topN) + ":" + str(final_recall))
        print("Reranking hole rate@10:" + str(hole_rate))
        print("Reranking hole rate:" + str(Ahole_rate))
        print("Reranking ms_mrr:" + str(ms_mrr))
