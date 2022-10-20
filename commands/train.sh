################################## Train_MSMARCO ################################
train_path='../data/datasets/MSMARCO/train_retrieval_large.json'
valid_path='../data/datasets/MSMARCO/train_retrieval_small.json'
query_embed_path='../data/datasets/MSMARCO/qembed.pkl'
train_doc_embed_path='../data/datasets/MSMARCO/embed/dembed.pkl'
output_dim=64
outdir='../data/datasets/MSMARCO/save_model_64'


train_cmd="\
python ../train/MSMARCO/train.py --train_path $train_path --valid_path $valid_path --query_embed_path $query_embed_path \
                             --train_doc_embed_path $train_doc_embed_path --output_dim $output_dim --outdir $outdir\
"

echo $train_cmd
eval $train_cmd
################################## Train_NQ ################################
train_path='../data/datasets/NQ/train_retrieval.json'
valid_path='../data/datasets/NQ/dev_retrieval.json'
query_embed_path='../data/datasets/NQ/qembed.pkl'
train_doc_embed_path='../data/datasets/NQ/dembed_train.pkl'
valid_doc_embed_path='../data/datasets/NQ/dembed_dev.pkl'
output_dim=64
outdir='../data/datasets/NQ/save_model_64'

train_cmd="\
python ../train/NQ/train.py --train_path $train_path --valid_path $valid_path --query_embed_path $query_embed_path \
                        --train_doc_embed_path $train_doc_embed_path --valid_doc_embed_path $valid_doc_embed_path \
                        --output_dim $output_dim --outdir $outdir\
"

echo $train_cmd
eval $train_cmd
