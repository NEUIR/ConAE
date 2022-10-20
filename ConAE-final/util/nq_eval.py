import logging

from util.dpr_utils import SimpleTokenizer, has_answer


logger = logging.getLogger(__name__)

def validate(passages, answers, closest_docs, query_embedding2id, passage_embedding2id):

    tok_opts = {}
    tokenizer = SimpleTokenizer(**tok_opts)

    logger.info('Matching answers in top docs...')
    scores = []
    for query_idx in range(closest_docs.shape[0]):
        query_id = query_embedding2id[query_idx]
        doc_ids = [passage_embedding2id[pidx] for pidx in closest_docs[query_idx]]
        hits = []
        for i, doc_id in enumerate(doc_ids):
            text = passages[doc_id]
            hits.append(has_answer(answers[query_id], text, tokenizer))
        scores.append(hits)

    logger.info('Per question validation results len=%d', len(scores))

    n_docs = len(closest_docs[0])
    top_k_hits = [0] * n_docs
    for question_hits in scores:
        best_hit = next((i for i, x in enumerate(question_hits) if x), None)
        if best_hit is not None:
            top_k_hits[best_hit:] = [v + 1 for v in top_k_hits[best_hit:]]

    logger.info('Validation results: top k documents hits %s', top_k_hits)
    top_k_hits = [v / len(closest_docs) for v in top_k_hits]
    logger.info('Validation results: top k documents hits accuracy %s', top_k_hits)
    return top_k_hits