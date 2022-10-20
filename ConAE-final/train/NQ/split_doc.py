import glob
import json
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

from util.data_loader import load_nq_queries, GenericDataLoader

logger = logging.getLogger(__name__)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    logger.info(args)
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.DEBUG,
                        datefmt='%d-%m-%Y %H:%M:%S')

    with open('../data/datasets/NQ/train_retrieval.json') as fin:
        dids = set()
        for step, line in tqdm(enumerate(fin)):
            example = json.loads(line.strip())
            golden_docs = example["golden_docs"]
            rank_docs = example["rank_docs"]
            for step, doc in enumerate(golden_docs):
                dids.add(doc)
            for step, doc in enumerate(rank_docs):
                    dids.add(doc[0])

    splits = glob.glob(os.path.join('../data/datasets/NQ',"dembed*.pkl"))
    output_file = os.path.join('../data/datasets/NQ', "dembed_train.pkl")
    print("Temperary validation files: {}".format(splits))
    res_dict = {}
    for s in splits:
        print(s)
        with open(s, 'rb') as f:
            data = pickle.load(f)
            for did in dids:
                if did in data.keys():
                    res_dict[did] = data[did]
    with open(output_file, 'wb') as f:
        pickle.dump(res_dict, f)
    print("Merge total {} lines".format(len(res_dict.keys())))



    with open('../data/datasets/NQ/dev_retrieval.json') as fin:
        dids = set()
        for step, line in tqdm(enumerate(fin)):
            example = json.loads(line.strip())
            golden_docs = example["golden_docs"]
            rank_docs = example["rank_docs"]
            for step, doc in enumerate(golden_docs):
                dids.add(doc)
            for step, doc in enumerate(rank_docs):
                    dids.add(doc[0])

    splits = glob.glob(os.path.join('../data/datasets/NQ',"dembed*.pkl"))
    output_file = os.path.join('../data/datasets/NQ', "dembed_dev.pkl")
    print("Temperary validation files: {}".format(splits))
    res_dict = {}
    for s in splits:
        print(s)
        with open(s, 'rb') as f:
            data = pickle.load(f)
            for did in dids:
                if did in data.keys():
                    res_dict[did] = data[did]
    with open(output_file, 'wb') as f:
        pickle.dump(res_dict, f)
    print("Merge total {} lines".format(len(res_dict.keys())))


