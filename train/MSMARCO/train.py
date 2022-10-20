import random,os
import argparse
import numpy as np
import torch
import torch.optim as optim
import logging
import pickle
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from data_loader import qdpair_data
from model import distill_model

logger = logging.getLogger(__name__)


def eval_model(model, validset, args):
    sampler = SequentialSampler(validset)
    dataloader = DataLoader(validset,
        sampler=sampler,
        batch_size=args.valid_batch_size,
        drop_last=False,
        num_workers=1,
        collate_fn=validset.collect_fn
    )
    model.eval()
    total_num = 0.0
    total_loss = 0.0
    with torch.no_grad():
        for step, batch_data in enumerate(dataloader):
            loss = model(batch_data["qembeds"].cuda(), batch_data["dembeds"].cuda(), batch_data["dscores"].cuda(), batch_data["dembeds_pos"].cuda(), batch_data["dembeds_neg"].cuda(), only_kl=True)
            loss = loss.item()
            total_loss += loss
            total_num += 1
    if total_num == 0:
        return 0.0
    return total_loss/total_num




def train_model(model, args, trainset_reader, validset):
    save_path = args.outdir + '/model'
    best_loss = float('inf')
    running_loss = 0.0

    optimizer = optim.Adam(model.parameters(), args.learning_rate)
    global_step = 0
    for epoch in range(int(args.num_train_epochs)):
        optimizer.zero_grad()
        for batch_data in trainset_reader:
            model.train()
            loss = model(batch_data["qembeds"].cuda(), batch_data["dembeds"].cuda(), batch_data["dscores"].cuda(),
                         batch_data["dembeds_pos"].cuda(), batch_data["dembeds_neg"].cuda(), only_kl=True)
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
                valid_loss = eval_model(model, validset, args)
                logger.info('Dev loss: {0}'.format(valid_loss))
                if valid_loss <= best_loss:
                    best_loss = valid_loss
                    torch.save({'epoch': epoch,
                                'model': model.state_dict()}, save_path + ".best.pt")
                    logger.info("Saved best epoch {0}, best loss {1}".format(epoch, best_loss))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', help='Train candidate path.')
    parser.add_argument('--valid_path', help='Valid candidate path.')
    parser.add_argument('--query_embed_path', help='Query embedding path.')
    parser.add_argument('--train_doc_embed_path', help='Document embedding path.')
    parser.add_argument('--outdir', help='Path to output directory')
    parser.add_argument("--train_batch_size", default=128, type=int, help="Total batch size for training.")
    parser.add_argument("--valid_batch_size", default=128, type=int, help="Total batch size for predictions.")
    parser.add_argument("--input_dim", default=768, type=int, help="Input dimension.")
    parser.add_argument("--output_dim", type=int, required=True, help="Output dimension.")
    parser.add_argument("--eval_step", default=1000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float,
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
    logger.info("loading validation set")
    validset = qdpair_data(args.valid_path, query_embeds, doc_embeds)
    del doc_embeds, query_embeds
    logger.info('initializing estimator model')
    model = distill_model(args).cuda()
    train_model(model, args, trainset_reader, validset)