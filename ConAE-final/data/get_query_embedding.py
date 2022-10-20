import argparse
import glob
import json
import logging
import os
import pickle
from typing import Optional
import torch.distributed as dist
import torch.utils.data.distributed
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import PreTrainedModel, RobertaConfig, RobertaModel
from transformers import RobertaTokenizer
from transformers.file_utils import is_torch_available, requires_backends

logger = logging.getLogger(__name__)

if is_torch_available():
    import torch


class AnceEncoder(PreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = 'ance_encoder'
    load_tf_weights = None
    _keys_to_ignore_on_load_missing = [r'position_ids']
    _keys_to_ignore_on_load_unexpected = [r'pooler', r'classifier']

    def __init__(self, config: RobertaConfig):
        requires_backends(self, 'torch')
        super().__init__(config)
        self.config = config
        self.roberta = RobertaModel(config)
        self.embeddingHead = torch.nn.Linear(config.hidden_size, 768)
        self.norm = torch.nn.LayerNorm(768)
        self.init_weights()

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def init_weights(self):
        self.roberta.init_weights()
        self.embeddingHead.apply(self._init_weights)
        self.norm.apply(self._init_weights)

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
    ):
        input_shape = input_ids.size()
        device = input_ids.device
        if attention_mask is None:
            attention_mask = (
                torch.ones(input_shape, device=device)
                if input_ids is None
                else (input_ids != self.roberta.config.pad_token_id)
            )
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output[:, 0, :]
        pooled_output = self.norm(self.embeddingHead(pooled_output))
        return pooled_output


class QueryData_MSMARCO(Dataset):
    def __init__(self, query_path, tokenizer):
        self.data = self.read_file(query_path)
        self.tokenizer = tokenizer

    def read_file(self, data_path):
        data = []
        with open(data_path) as fin:
            for line in fin:
                example = json.loads(line.strip())
                data.append(example)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def collect_fn(self, data):
        qids = []
        queries = []
        for example in data:
            qids.append(example["_id"])
            queries.append(example["text"])
        inputs = self.tokenizer(
            queries,
            max_length=64,
            padding='longest',
            truncation=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        return {
            "qids": qids,
            "queries": queries,
            "query_tokens": inputs["input_ids"]
        }


class QueryData_NQ(Dataset):
    def __init__(self, query_path, tokenizer):
        self.data = self.read_file(query_path)
        self.tokenizer = tokenizer

    def read_file(self, data_paths):
        data = []
        for data_path in data_paths:
            with open(data_path) as fin:
                for line in fin:
                    example = json.loads(line.strip())
                    data.append(example)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def collect_fn(self, data):
        qids = []
        queries = []
        for example in data:
            qids.append(example["qid"])
            queries.append(example["text"])
        inputs = self.tokenizer(
            queries,
            max_length=64,
            padding='longest',
            truncation=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        return {
            "qids": qids,
            "queries": queries,
            "query_tokens": inputs["input_ids"]
        }


def merge_files(split_pattern, output_file):
    splits = glob.glob(split_pattern)
    print("Temperary validation files: {}.".format(splits))
    res_dict = {}
    for s in splits:
        with open(s, 'rb') as f:
            data = pickle.load(f)
            res_dict.update(data)
        os.remove(s)
    with open(output_file, 'wb') as f:
        pickle.dump(res_dict, f)
    print("Merge total {} lines.".format(len(res_dict.keys())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help="Dataset name like MSMARCO, NQ.")
    parser.add_argument('--tokenizer', type=str, metavar='name or path', required=True,
                        help="Path to a hgf tokenizer name or path.")
    parser.add_argument('--encoder', type=str, required=True,
                        help="Path to an encoder name or path.")
    parser.add_argument('--data_path', type=str, required=True, help="Path to datasets.")
    parser.add_argument('--output_path', type=str, metavar='path', required=True, help="Path to output file.")
    parser.add_argument('--batch_size', type=int, required=False, default=128, help="Batch size.")
    parser.add_argument('--local_rank', type=int, default=-1, help="For distributed training: local_rank.")
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
    logger.info(args)

    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)

    logger.info('Loading ANCE model!')
    query_encoder = AnceEncoder.from_pretrained(args.encoder).eval()
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer)

    if args.local_rank != -1:
        query_encoder = torch.nn.parallel.DistributedDataParallel(
            query_encoder.to(device),
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True
        )
    if args.dataset.lower() == 'msmarco':
        query_data = QueryData_MSMARCO(os.path.join(args.data_path, "queries.jsonl"), tokenizer)
    elif args.dataset.lower() == 'nq':
        data_paths = ["nq-dev.json", "nq-train.json", "nq-test.json"]
        data_paths = [os.path.join(args.data_path, path) for path in data_paths]
        query_data = QueryData_NQ(data_paths, tokenizer)

    query_sampler = torch.utils.data.distributed.DistributedSampler(query_data)
    query_loader = DataLoader(query_data, batch_size=args.batch_size, sampler=query_sampler,
                              collate_fn=query_data.collect_fn)
    qembed_path = os.path.join(args.output_path, "q_embed{0}.pkl".format(args.local_rank))
    query_embed_dict = {}

    logger.info('Encoding query!')
    for batch in tqdm(query_loader):
        query_embeds = query_encoder(batch["query_tokens"]).detach().cpu().numpy()
        query_ids = batch["qids"]
        assert len(query_ids) == len(query_embeds)
        for qid, qembed in zip(query_ids, query_embeds):
            query_embed_dict[qid] = qembed

    logger.info('Saving query file!')
    with open(qembed_path, "wb") as f_qembed:
        pickle.dump(query_embed_dict, f_qembed)
    dist.barrier()
    if args.local_rank == 0:
        merge_files(os.path.join(args.output_path, "q_embed*.pkl"), os.path.join(args.output_path, "qembed.pkl"))
