import json
import numpy as np
import torch.utils.data.distributed
from torch.utils.data import Dataset


class qdpair_data(Dataset):
    def __init__(self, data_path, query_embeds, doc_embeds):
        self.query_embeds = query_embeds
        self.doc_embeds = doc_embeds
        self.read_file(data_path)

    def read_file(self, data_path):
        self.data = []
        with open(data_path) as fin:
            for step, line in enumerate(fin):
                example = json.loads(line.strip())
                if len(example["golden_docs"]) > 0:
                    dids = []
                    dscores = []
                    neg_dids = []
                    qid = example["qid"]
                    golden_docs = example["golden_docs"]
                    rank_docs = example["rank_docs"][:100]
                    did = golden_docs[0]
                    np.random.shuffle(rank_docs)
                    golden_docs_dict = set(golden_docs)
                    for step, doc in enumerate(rank_docs):
                        if len(neg_dids) < 1 and doc[0] not in golden_docs_dict:
                            neg_dids.append(doc[0])
                        dids.append(doc[0])
                        dscores.append(doc[1])
                    qd_pairs = (qid, did, dids, dscores, neg_dids)
                    self.data.append(qd_pairs)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def collect_fn(self, batch_data):
        batch_dscores = []
        batch_qembeds = []
        batch_dembeds = []
        batch_dembeds_pos = []
        batch_dembeds_neg = []

        for example in batch_data:
            qid, did, dids, dscores, neg_dids = example
            batch_qembeds.append(self.query_embeds[qid])
            batch_dembeds_pos.append(self.doc_embeds[did])
            for did in dids:
                batch_dembeds.append(self.doc_embeds[did])
            for did in neg_dids:
                batch_dembeds_neg.append(self.doc_embeds[did])
            batch_dscores.extend(dscores)

        return {
            "dembeds_pos": torch.tensor(np.array(batch_dembeds_pos)).float(),
            "dembeds_neg": torch.tensor(np.array(batch_dembeds_neg)).float(),
            "dscores": torch.tensor(np.array(batch_dscores)).float(),
            "qembeds": torch.tensor(np.array(batch_qembeds)).float(),
            "dembeds": torch.tensor(np.array(batch_dembeds)).float()
        }