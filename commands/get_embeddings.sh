################################## Compress ################################
dataset="Dataset name like MSMARCO, TREC_DL, TREC_COVID."
query_embed_path="Query embedding path"
doc_embed_path="Document embedding path"
checkpoint="Checkpoint path"
output_dim=64
output_path="Compressed embedding path"

compress_cmd="\
python ../get_embbedings/get_emb.py --dataset $dataset --query_embed_path $query_embed_path --doc_embed_path $doc_embed_path \
                             --checkpoint $checkpoint --output_dim $output_dim --output_path $output_path\
"

echo $compress_cmd
eval $compress_cmd

################################## Compress_NQ ################################
#query_embed_path="Query embedding path"
#doc_embed_path="Document embedding path"
#checkpoint="Checkpoint path"
#nq_test_path="NQ test dataset queries and answers path."
#output_dim=64
#output_path="Compressed embedding path"
#
#compress_cmd="\
#python ../get_embbedings/get_nq_emb.py --query_embed_path $query_embed_path --doc_embed_path $doc_embed_path \
#                             --checkpoint $checkpoint --nq_test_path $nq_test_path \
#                             --output_dim $output_dim --output_path $output_path
#"
#
#echo $compress_cmd
#eval $compress_cmd
