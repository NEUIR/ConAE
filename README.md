# ConAE

This repo provides the code for reproducing the experiments in [Dimension Reduction for Efficient Dense Retrieval via Conditional Autoencoder](https://arxiv.org/pdf/2205.03284.pdf)

## Requirements

To install requirements, run the following commands:

```
git clone https://github.com/NEUIR/ConAE/
cd ConAE
python setup.py install
```
## Data Download

To download all the needed data, run:
```
bash commands/data_download.sh 
```
## Original embeddings generation
All embeddings come from the dataset encoded by the [ANCE](https://github.com/microsoft/ANCE) encoder in the format of 768-dimensional dense vectors. You can encode the dataset yourself, or download the already encoded embeddings directly from other repositories such as [Pyserini](https://github.com/castorini/pyserini) and [Beir](https://github.com/beir-cellar/beir). If you generate the embedding yourself or download it from other places, pay attention to the format of this repository, otherwise an error will occur. The dataset embeddings we use are given below.

1.To encode MSMARCO or TREC_COVID dataset, run:
```
python ../encode/get_dense_emb.py \
  --dataset MSMARCO \
  --output_path ../data/datasets/msmarco/embbedings/
```

2.To encode TREC_DL dataset, run:
```
python ../encode/get_trecdl_emb.py \
  --query_path ../data/datasets/trec_dl/2019qrels-pass.txt \
  --output_path ../data/datasets/trec_dl/embbedings
```

3.To encode NQ dataset, please download from [Pyserini](https://rgw.cs.uwaterloo.ca/JIMMYLIN-bucket0/pyserini-indexes/dindex-wikipedia-ance_multi-bf-20210224-060cef.tar.gz).

## Compress embeddings 

1.To Compress dense embeddings(MSMARCO, TREC_DL, TREC_COVID) using ConAE, run:
```
python ../get_embbedings/get_emb.py \
  --dataset MSMARCO \
  --query_embed_path ../data/datasets/msmarco/embbedings/qembed.pkl \
  --doc_embed_path ../data/datasets/msmarco/embbedings/dembed.pkl \
  --checkpoint ../data/checkpoint/ConAE/marco/ConAE_256/model.best.pt  \
  --output_dim 256 \
  --output_path ../data/compress_emb/msmarco/ 
```

2.To Compress dense embeddings(NQ) using ConAE, run:
```
python ../get_embbedings/get_nq_emb.py \
  --query_embed_path ../data/datasets/nq/query-embedding-ance_multi-nq-test/ \
  --doc_embed_path ../data/datasets/nq/dindex-wikipedia-ance_multi-bf-20210224-060cef/ \
  --checkpoint ../data/checkpoint/ConAE/NQ/ConAE_256/model.best.pt \
  --nq_test_path ../data/datasets/nq/nq-test.json \
  --output_dim 256 \
  --output_path ../data/compress_emb/nq/ 
```

## Evaluation

The command for evaluation is the same as that for Compress embeddings described above. However you need to add --evaluation to the command to have the program to evaluate after the compress embeddings step. commands/run_inference.sh provides a sample command.


## Results

| MSMARCO | MRR@10	| NDCG@10| Recall@1k |
| ---- | -------- | ----- | ------- | 
| ANCE-768  | 0.3302     | 0.3877  | 0.9584    |
| ConAE-256  | 0.3294     | 0.3864  | 0.9560    |
| ConAE-128  | 0.3245     | 0.3816  | 0.9523    |
| ConAE-64  | 0.2862     | 0.3376  | 0.9222    |

| NQ  | Top20	| Top100|                                          
| ---- | -------- | ----- | 
| ANCE-768  | 0.8224     | 0.8787  |
| ConAE-256  | 0.8053     | 0.8723 |
| ConAE-128  | 0.8064     | 0.8687 |
| ConAE-64  | 0.7604 | 0.8460 |

| TREC-COVID | NDCG@10|
| ---- | -------- | 
| ANCE-768  | 0.6529     | 
| ConAE-256  | 0.6405    | 
| ConAE-128  | 0.6381    | 
| ConAE-64  | 0.5006     | 

Our checkpoints could be downloaded [here](https://1drv.ms/u/s!AlOd75jmn2v0gQeMGXxBFlQf8Kwx?e=EzYTcw). 
