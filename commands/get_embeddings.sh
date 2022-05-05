##################################### Get MSMARCO&TRECCOVID NEmbeddings ################################
#output_path="../data/datasets/msmarco/embbedings/emb"
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


emb_cmd="\
python ../encode/get_trecdl_emb.py --output_path $output_path\
"

echo $emb_cmd
eval $emb_cmd