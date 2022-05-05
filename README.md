# ConAE

This repo provides the code for reproducing the experiments in [Dimension Reduction for Efficient Dense Retrieval via Conditional Autoencoder]()

## Requirements

To install requirements, run the following commands:

```
git clone https://github.com/NEUIR/ConAE/
cd ConAE
python setup.py install
```
## Data Download

All embeddings come from the dataset encoded by the [ANCE](https://github.com/microsoft/ANCE) encoder in the format of 768-dimensional dense vectors.You can encode the dataset yourself, or download the already encoded embeddings directly.The dataset embeddings we use are given below. If you generate the embedding yourself or download it from other places, pay attention to the format of this repository, otherwise an error will occur.

| Datasets   | Getting embedded method                                      |
| ---------- | :----------------------------------------------------------- |
| MS MARCO   | bash commands/get_embeddings.sh                              |
| TREC_DL    | bash commands/get_embeddings.sh                              |
| TREC_COVID | bash commands/get_embeddings.sh                              |
| NQ         | https://rgw.cs.uwaterloo.ca/JIMMYLIN-bucket0/pyserini-indexes/dindex-wikipedia-ance_multi-bf-20210224-060cef.tar.gz |

## Evaluation

The evaluation is done through the package of "evaluation".You can run the bash file to evaluate.In order to run it, you need to define the following parameters at the beginning of the eval_*.py.

```
dataset="Dataset name like MSMARCO, TREC_DL, NQ, TREC_COVID."
query_embed_path="Query embedding path"
doc_embed_path="Document embedding path"
checkpoint="Checkpoint path or PCA model path"
model="Model name like ConAE, KL, CE, PCA"
output_dim="Output dimension"

eval_cmd="\
python ../evaluation/eval_msmarco.py --dataset  $dataset --query_embed_path $query_embed_path --doc_embed_path $doc_embed_path \
                             --checkpoint $checkpoint --model $model --output_dim $output_dim\
"

echo $eval_cmd
eval $eval_cmd
```
