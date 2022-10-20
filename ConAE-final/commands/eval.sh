################################## Compress_MSMARCO ################################
query_embed_path='../data/datasets/MSMARCO/qembed.pkl'
doc_embed_path='../data/datasets/MSMARCO/dembed.pkl'
checkpoint='../data/datasets/MSMARCO/save_model_256/model.best.pt'
output_dim=256
output_path='../data/datasets/MSMARCO/compress_emb/'

compress_cmd="\
python ../get_embbedings/get_emb.py --query_embed_path $query_embed_path --doc_embed_path $doc_embed_path \
                             --checkpoint $checkpoint --output_dim $output_dim --output_path $output_path --evaluate\
"

echo $compress_cmd
eval $compress_cmd

################################## Compress_NQ ################################
query_embed_path="../data/datasets/NQ/qembed.pkl"
doc_embed_path="../data/datasets/NQ/dindex-wikipedia-ance_multi-bf-20210224-060cef"
checkpoint='../data/datasets/NQ/save_model_256/model.best.pt'
output_dim=256
output_path='../data/datasets/NQ/compress_emb/'


compress_cmd="\
python ../get_embbedings/get_nq_emb.py --query_embed_path $query_embed_path --doc_embed_path $doc_embed_path \
                             --checkpoint $checkpoint --output_dim $output_dim --output_path $output_path --evaluate\
"

echo $compress_cmd
eval $compress_cmd
