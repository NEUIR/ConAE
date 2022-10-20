####################################### Get MSMARCO Doc Embeddings ################################
#index='../data/datasets/MSMARCO/dindex-msmarco-passage-ance-bf-20210224-060cef'
#output_path='../data/datasets/MSMARCO'
#partition=1
#
#emb_cmd="\
#python ../data/get_doc_embedding.py --index $index --output_path $output_path --partition $partition\
#"
#
#echo $emb_cmd
#eval $emb_cmd
#
###################################### Get MSMARCO Query Embeddings ################################
#dataset='MSMARCO'
#tokenizer='castorini/ance-msmarco-passage'
#encoder='castorini/ance-msmarco-passage'
#output_path='../data/datasets/MSMARCO'
#data_path='../data/datasets/MSMARCO'
#
#emb_cmd="\
#python -u -m torch.distributed.launch ../data/get_query_embedding.py --dataset $dataset --tokenizer $tokenizer \
#                                    --encoder $encoder --output_path $output_path --data_path $data_path \
#"
#
#echo $emb_cmd
#eval $emb_cmd

###################################### Get NQ Doc Embeddings ################################
#index='../data/datasets/NQ/dindex-wikipedia-ance_multi-bf-20210224-060cef'
#output_path='../data/datasets/NQ'
#partition=10
#
#emb_cmd="\
#python ../data/get_doc_embedding.py --index $index --output_path $output_path --partition $partition\
#"
#
#echo $emb_cmd
#eval $emb_cmd



###################################### Get NQ Query Embeddings ################################
dataset='NQ'
tokenizer='castorini/ance-dpr-question-multi'
encoder='castorini/ance-dpr-question-multi'
output_path='../data/datasets/NQ'
data_path='../data/datasets/NQ'

emb_cmd="\
python -u -m torch.distributed.launch ../encode/NQ/get_query_embedding.py --dataset $dataset --tokenizer $tokenizer \
                                    --encoder $encoder --output_path $output_path --data_path $data_path \
"

echo $emb_cmd
eval $emb_cmd