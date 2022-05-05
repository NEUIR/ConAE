###################################### Eval ################################
#dataset="Dataset name like MSMARCO, TREC_DL."
#query_embed_path="Query embedding path"
#doc_embed_path="Document embedding path"
#checkpoint="Checkpoint path or PCA model path"
#model="Model name like ConAE, KL, CE, PCA"
#output_dim="Output dimension"
#
#eval_cmd="\
#python ../evaluation/eval_msmarco.py --dataset  $dataset --query_embed_path $query_embed_path --doc_embed_path $doc_embed_path \
#                             --checkpoint $checkpoint --model $model --output_dim $output_dim\
#"
#
#echo $eval_cmd
#eval $eval_cmd

