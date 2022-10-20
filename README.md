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
All embeddings come from the dataset encoded by the [ANCE](https://github.com/microsoft/ANCE) encoder in the format of 768-dimensional dense vectors. You can encode the dataset yourself, or download the already encoded embeddings directly from other repositories such as [Pyserini](https://github.com/castorini/pyserini). If you generate the embedding yourself or download it from other places, pay attention to the format of this repository, otherwise an error will occur. The dataset embeddings we use are given below.

1.To encode queries, run:
```
python -u -m torch.distributed.launch ../data/get_query_embedding.py \
	--dataset MSMARCO
	--tokenizer castorini/ance-msmarco-passage \
	--encoder castorini/ance-msmarco-passage \
	--output_path ../data/datasets/MSMARCO \
	--data_path ../data/datasets/MSMARCO \
```
2.To encode docs, run:

```
python ../data/get_doc_embedding.py \
	--index ../data/datasets/MSMARCO/dindex-msmarco-passage-ance-bf-20210224-060cef \
	--output_path ../data/datasets/MSMARCO	\
	--partition 1
```

## Training
### MSMARCO
1.To train the ConAE in the paper for MSMARCO, you need to get candidates before training:
```
python ../train/MSMARCO/get_candidates.py 
	--index ../data/datasets/MSMARCO/dindex-msmarco-passage-ance-bf-20210224-060ce \
	--qembed_path ../data/datasets/MSMARCO/qembed.pkl \
	--output_path ../data/datasets/MSMARCO \
	--topk 500 --batch_size 128 --threads 40
```
2.Then, we randomly sample 50,000 queries from the raw training set of MSMARCO as the development set.
```
python ../train/MSMARCO/split_train.py
```
3.Finally, run:
```
python ../train/MSMARCO/train.py 
	--train_path ../data/datasets/MSMARCO/train_retrieval_large.json \
	--valid_path ../data/datasets/MSMARCO/train_retrieval_small.json \
	--query_embed_path ../data/datasets/MSMARCO/qembed.pkl \
	--train_doc_embed_path ../data/datasets/MSMARCO/embed/dembed.pkl \
	--output_dim 64 \
	--outdir ../data/datasets/MSMARCO/save_model_64
```
### NQ
1.To train the ConAE in the paper for NQ, you need to get candidates before training:
```
python ../train/NQ/get_candidates.py 
	--index ../data/datasets/NQ/dindex-wikipedia-ance_multi-bf-20210224-060cef \
	--qembed_path ../data/datasets/NQ/qembed.pkl \
	--output_path ../data/datasets/NQ \
	--topk 500 --threads 40
```
2.Because of memory limitation, we divide docs into dev sets and train sets. If you have enough memory, you can omit this step.
```
python ../train/NQ/split_doc.py
```
3.Finally, run:
```
python ../train/NQ/train.py 
	--train_path ../data/datasets/NQ/train_retrieval.json \
	--valid_path ../data/datasets/NQ/dev_retrieval.json \
	--query_embed_path ../data/datasets/NQ/qembed.pkl \
	--train_doc_embed_path ../data/datasets/NQ/dembed_train.pkl \
	--valid_doc_embed_path ../data/datasets/NQ/dembed_dev.pkl \
	--output_dim 64 \
	--outdir ../data/datasets/NQ/save_model_64
```

## Compress embeddings 

1.To Compress dense embeddings(MSMARCO) using ConAE, run:
```
python ../get_embbedings/get_emb.py \
  --query_embed_path ../data/datasets/MSMARCO/qembed.pkl \
  --doc_embed_path ../data/datasets/MSMARCO/embed/dembed.pkl \
  --checkpoint ../data/checkpoint/ConAE/MSMARCO/ConAE_256/model.best.pt  \
  --output_dim 256 \
  --output_path ../data/datasets/MSMARCO/compress_emb/
```

2.To Compress dense embeddings(NQ) using ConAE, run:
```
python ../get_embbedings/get_nq_emb.py \
  --query_embed_path ../data/datasets/NQ/query-embedding-ance_multi-nq-test/ \
  --doc_embed_path ../data/datasets/NQ/dindex-wikipedia-ance_multi-bf-20210224-060cef/ \
  --checkpoint ../data/checkpoint/ConAE/NQ/ConAE_256/model.best.pt \
  --output_dim 256 \
  --output_path ../data/datasets/NQ/compress_emb/
```

Our checkpoints could be downloaded [here](https://1drv.ms/u/s!AlOd75jmn2v0gQeMGXxBFlQf8Kwx?e=EzYTcw). 

## Evaluation

The command for evaluation is the same as that for Compress embeddings described above. However you need to add --evaluation to the command to have the program to evaluate after the compress embeddings step.


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
