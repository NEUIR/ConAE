import argparse
import logging
import os
import pickle

import faiss
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--partition', type=int, required=False, default=1, help="Number of partition.")
    parser.add_argument('--index', type=str, required=True, help="Path to prebuilt index.")
    parser.add_argument('--output_path', type=str, metavar='path', required=True, help="Path to output file.")
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
    logger.info(args)

    logger.info('Loading prebuilt index!')
    index_path = os.path.join(args.index, 'index')
    docid_path = os.path.join(args.index, 'docid')
    docindex = faiss.read_index(index_path)
    dids = []
    with open(docid_path, 'r') as fin:
        for line in fin:
            dids.append(line.strip())

    logger.info('Spliting and saving doc files!')
    batch_size = int(np.ceil(len(dids) * 1.0 / args.partition))
    patition_counter = 0
    for strat_id in tqdm(range(0, len(dids), batch_size)):
        doc_embeds = {}
        for step in range(strat_id, min(strat_id + batch_size, len(dids))):
            doc_embedding = docindex.reconstruct(step)
            doc_embeds[dids[step]] = np.array(doc_embedding)
        dembed_path = os.path.join(args.output_path, "dembed_{0}.pkl".format(patition_counter))
        with open(dembed_path, "wb") as f_dembed:
            pickle.dump(doc_embeds, f_dembed)
        patition_counter += 1
