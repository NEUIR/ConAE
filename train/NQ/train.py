import random, os
import argparse
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from model import distill_model
from data_loader import qdpair_data, qdpair_data_nq
import logging
import pickle
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

logger = logging.getLogger(__name__)





def validate(answers, qids, closest_docs):
    logger.info('Matching answers in top docs...')
    scores = []
    assert len(qids) == len(closest_docs)
    for i in range(len(qids)):
        doc_ids = closest_docs[i]
        query_id = qids[i]
        hits = []
        for doc_id in doc_ids:
            if doc_id in answers[query_id]:
                hits.append(True)
            else:
                hits.append(False)
        scores.append(hits)

    logger.info('Per question validation results len=%d', len(scores))

    n_docs = len(closest_docs[0])
    top_k_hits = [0] * n_docs
    for question_hits in scores:
        best_hit = next((i for i, x in enumerate(question_hits) if x), None)
        if best_hit is not None:
            top_k_hits[best_hit:] = [v + 1 for v in top_k_hits[best_hit:]]
    top_k_hits = [v / len(closest_docs) for v in top_k_hits]
    return top_k_hits[19]

def eval_model(model, validset, args):
    sampler = SequentialSampler(validset)
    dataloader = DataLoader(validset,
        sampler=sampler,
        batch_size=args.valid_batch_size,
        drop_last=False,
        num_workers=10,
        collate_fn=validset.collect_fn
    )
    model.eval()

    rank_scores = {}
    rank_list = []
    qid_list = []
    with torch.no_grad():
        for step, batch_data in tqdm(enumerate(dataloader)):
            scores = model.get_scores(batch_data["qembeds"].cuda(), batch_data["dembeds"].cuda())
            qids = batch_data["qids"]
            dids = batch_data["dids"]
            assert len(scores) == len(qids) == len(dids)
            for qid, did, score in list(zip(qids, dids, scores)):
                if qid not in rank_scores:
                    rank_scores[qid] = []
                rank_scores[qid].append([did, score])
    for qid, ranks in tqdm(rank_scores.items()):
        doc_ranks = sorted(ranks, key=lambda x:x[1], reverse=True)
        doc_ranks = [doc[0] for doc in doc_ranks][:100]
        rank_list.append(doc_ranks)
        qid_list.append(qid)
    acc = validate(validset.answer_dict, qid_list, rank_list)

    return acc




def train_model(model, args, trainset_reader, validset):
    save_path = args.outdir + '/model'
    best_acc = 0.0
    running_loss = 0.0

    optimizer = optim.Adam(model.parameters(), args.learning_rate)
    global_step = 0
    for epoch in range(int(args.num_train_epochs)):
        optimizer.zero_grad()
        for batch_data in trainset_reader:
            model.train()
            loss = model(batch_data["qembeds"].cuda(), batch_data["dembeds"].cuda(), batch_data["dscores"].cuda(),
                         batch_data["dembeds_pos"].cuda(), batch_data["dembeds_neg"].cuda())
            running_loss += loss.item()
            if args.gradient_accumulation_steps != 0:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            global_step += 1
            if global_step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                logger.info('Epoch: {0}, Step: {1}, Loss: {2}'.format(epoch, global_step, (running_loss / global_step)))
            if global_step % (args.eval_step * args.gradient_accumulation_steps) == 0:
                logger.info('Start eval!')
                valid_acc = eval_model(model, validset, args)
                logger.info('Dev acc: {0}'.format(valid_acc))
                if valid_acc >= best_acc:
                    best_acc = valid_acc
                    torch.save({'epoch': epoch,
                                'model': model.state_dict()}, save_path + ".best.pt")
                    logger.info("Saved best epoch {0}, best acc {1}".format(epoch, best_acc))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', help='Train candidate path.')
    parser.add_argument('--valid_path', help='Valid candidate path.')
    parser.add_argument('--query_embed_path', help='Query embedding path.')
    parser.add_argument('--train_doc_embed_path', help='Document embedding path.')
    parser.add_argument('--valid_doc_embed_path', help='Document embedding path')
    parser.add_argument('--outdir', help='Path to output directory')
    parser.add_argument("--train_batch_size", default=128, type=int, help="Total batch size for training.")
    parser.add_argument("--valid_batch_size", default=512, type=int, help="Total batch size for predictions.")
    parser.add_argument("--input_dim", default=768, type=int, help="Total batch size for training.")
    parser.add_argument("--output_dim", type=int, required=True, help="Output dimension.")
    parser.add_argument("--eval_step", default=1000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=20.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    args = parser.parse_args()
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
    handlers = [logging.FileHandler(os.path.abspath(args.outdir) + '/train_log.txt'), logging.StreamHandler()]
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.DEBUG,
                        datefmt='%d-%m-%Y %H:%M:%S', handlers=handlers)
    logger.info(args)

    logger.info('Start training!')
    logger.info("loading query embeddings")
    with open(args.query_embed_path, "rb") as fin:
        query_embeds = pickle.load(fin)

    logger.info("loading doc embeddings")
    with open(args.train_doc_embed_path, "rb") as fin:
        doc_embeds = pickle.load(fin)

    logger.info("loading training set")
    trainset = qdpair_data(args.train_path, query_embeds, doc_embeds)
    train_sampler = RandomSampler(trainset)
    trainset_reader = DataLoader(trainset,
                                 sampler=train_sampler,
                                 batch_size=args.train_batch_size,
                                 num_workers=10,
                                 collate_fn=trainset.collect_fn)
    logger.info("loading doc embeddings")
    with open(args.valid_doc_embed_path, "rb") as fin:
        doc_embeds = pickle.load(fin)
    logger.info("loading validation set")
    validset = qdpair_data_nq(args.valid_path, query_embeds, doc_embeds)

    logger.info('initializing estimator model')
    model = distill_model(args).cuda()
    train_model(model, args, trainset_reader, validset)
