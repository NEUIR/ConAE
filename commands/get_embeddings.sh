##################################### Get MSMARCO&TRECCOVID Embeddings ################################
#dataset="Dataset name like MSMARCO, TREC_DL, TREC_COVID."
#query_embed_path="Query embedding path"
#doc_embed_path="Document embedding path"
#output_path="Compressed embedding path"
#checkpoint="Checkpoint path"
#output_dim="Output dimension"
#
#cmd="\
#python ../evaluation/eval_msmarco.py --dataset  $dataset --query_embed_path $query_embed_path --doc_embed_path $doc_embed_path \
#                             --output_path $output_path --checkpoint $checkpoint --output_dim $output_dim\
#"
#
#echo $cmd
#eval $cmd

