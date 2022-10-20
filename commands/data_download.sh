mkdir ../data/datasets/MSMARCO
cd ../data/datasets/MSMARCO

# download MSMARCO data
wget https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip
unzip msmarco.zip
rm msmarco.zip

wget https://rgw.cs.uwaterloo.ca/JIMMYLIN-bucket0/pyserini-indexes/dindex-msmarco-passage-ance-bf-20210224-060cef.tar.gz
tar -zxvf dindex-msmarco-passage-ance-bf-20210224-060cef.tar.gz
rm dindex-msmarco-passage-ance-bf-20210224-060cef.tar.gz


mkdir ../data/datasets/NQ
cd ../data/datasets/NQ

# download NQ data
wget https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip
unzip nq.zip
rm nq.zip

wget https://github.com/castorini/pyserini-data/raw/main/encoded-queries/query-embedding-ance_multi-nq-test-20210419-9323ec.tar.gz
tar -zxvf query-embedding-ance_multi-nq-test-20210419-9323ec.tar.gz
rm query-embedding-ance_multi-nq-test-20210419-9323ec.tar.gz

wget https://rgw.cs.uwaterloo.ca/JIMMYLIN-bucket0/pyserini-indexes/dindex-wikipedia-ance_multi-bf-20210224-060cef.tar.gz
tar -zxvf dindex-wikipedia-ance_multi-bf-20210224-060cef.tar.gz
rm dindex-wikipedia-ance_multi-bf-20210224-060cef.tar.gz

wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
tar -zxvf psgs_w100.tsv.gz
rm psgs_w100.tsv.gz