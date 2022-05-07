##################################### Get MSMARCO&TRECCOVID Embeddings ################################
#output_path="../data/datasets/msmarco/embbedings/"
#dataset=MSMARCO
#
#
#emb_cmd="\
#python ../encode/get_dense_emb.py --output_path $output_path\
#"
#
#echo $emb_cmd
#eval $emb_cmd

#################################### Get trecdl Embeddings ################################
output_path="../data/datasets/trec_covid/embbedings"
query_path="../data/datasets/trec_covid/2019qrels-pass.txt"

emb_cmd="\
python ../encode/get_trecdl_emb.py --output_path $output_path\
"

echo $emb_cmd
eval $emb_cmd
