##################################### Get MSMARCO & TRECCOVID Embeddings ################################
dataset=MSMARCO
output_path="../data/datasets/msmarco/embbedings/"

emb_cmd="\
python ../encode/get_dense_emb.py --dataset $dataset --output_path $output_path\
"

echo $emb_cmd
eval $emb_cmd
##################################### Get trecdl Embeddings ################################
#output_path="../data/datasets/trec_dl/embbedings"
#query_path="../data/datasets/trec_dl/2019qrels-pass.txt"
#
#emb_cmd="\
#python ../encode/get_trecdl_emb.py --query_path $query_path --output_path $output_path\
#"
#
#echo $emb_cmd
#eval $emb_cmd

