import numpy as np

data = list()
np.random.seed(1234)
with open("../data/datasets/MSMARCO/train_retrieval.json") as fin:
    for line in fin:
        data.append(line)
np.random.shuffle(data)

with open("../data/datasets/MSMARCO/train_retrieval_large.json", "w") as fout1, \
        open("../data/datasets/MSMARCO/train_retrieval_small.json", "w") as fout2:
    for step, line in enumerate(data):
        if step < 50000:
            fout2.write(line)
        else:
            fout1.write(line)