##################################### Get MSMARCO candidates ################################
index='../data/datasets/MSMARCO/dindex-msmarco-passage-ance-bf-20210224-060cef'
qembed_path='../data/datasets/MSMARCO/qembed.pkl'
output_path='../data/datasets/MSMARCO'
topk=500
batch_size=128
threads=40

candidates_cmd="\
python ../train/MSMARCO/get_candidates.py --index $index --qembed_path $qembed_path --output_path $output_path \
                                    --topk $topk --batch_size $batch_size --threads $threads\
"

echo $candidates_cmd
eval $candidates_cmd

###################################### Get NQ candidates ################################
index='../data/datasets/NQ/dindex-wikipedia-ance_multi-bf-20210224-060cef'
qembed_path='../data/datasets/NQ/qembed.pkl'
output_path='../data/datasets/NQ'
data_path='../data/datasets/NQ'
topk=500
threads=40


candidates_cmd="\
python ../train/NQ/get_candidates.py --index $index --qembed_path $qembed_path --output_path $output_path \
                                    --data_path $data_path --topk $topk --threads $threads\
"

echo $candidates_cmd
eval $candidates_cmd
