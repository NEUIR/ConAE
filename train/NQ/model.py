import torch
import torch.nn as nn
import torch.nn.functional as F


class distill_model(nn.Module):
    def __init__(self, args):
        super(distill_model, self).__init__()
        self.input_dim = args.input_dim
        self.output_dim = args.output_dim
        self.query_encoder = nn.Linear(self.input_dim, self.output_dim)
        self.doc_encoder = nn.Linear(self.input_dim, self.output_dim)
        self.projector = nn.Linear(args.output_dim, args.input_dim)
        self.loss_fct = torch.nn.KLDivLoss(reduction="batchmean")
        self.tanh = nn.Tanh()

    def enocde_queries(self, query_embeds):
        query_embeds = self.query_encoder(query_embeds)
        return query_embeds

    def enocde_docs(self, doc_embeds):
        doc_embeds = self.doc_encoder(doc_embeds)
        return doc_embeds

    def kldivloss(self, score, gold_score):
        gold_score = torch.softmax(gold_score, dim=-1)
        score = torch.nn.functional.log_softmax(score, dim=-1)
        return self.loss_fct(score, gold_score)

    def get_scores(self, query_embeds, doc_embeds):
        query_embeds = self.query_encoder(query_embeds)
        batch_size, embedding_dim = query_embeds.size()
        query_embeds = query_embeds.view(batch_size, 1, embedding_dim)
        doc_embeds = self.doc_encoder(doc_embeds)
        doc_embeds = doc_embeds.view(batch_size, -1, embedding_dim)
        scores = torch.sum(query_embeds * doc_embeds, -1)
        return scores

    def forward(self, query_embeds, doc_embeds, golden_scores, doc_embeds_pos, doc_embeds_neg, only_kl=False):
        batch_size = query_embeds.size(0)
        raw_query_embeds = query_embeds.view(batch_size, 1, self.input_dim)
        raw_doc_embeds = doc_embeds.view(batch_size, -1, self.input_dim)
        query_embeds = self.query_encoder(raw_query_embeds)
        doc_embeds = self.doc_encoder(raw_doc_embeds)
        scores = torch.bmm(query_embeds, torch.transpose(doc_embeds, 1, 2)).squeeze(1)
        loss1 = self.kldivloss(scores.view(batch_size, -1), golden_scores.view(batch_size, -1))
        if only_kl:
            return loss1
        raw_doc_embeds_pos = doc_embeds_pos.view(batch_size, 1, self.input_dim)
        raw_doc_embeds_neg = doc_embeds_neg.view(batch_size, -1, self.input_dim)
        doc_embeds_pos = self.projector(self.doc_encoder(raw_doc_embeds_pos))
        doc_embeds_neg = self.projector(self.doc_encoder(raw_doc_embeds_neg))
        query_embeds = self.projector(query_embeds)
        pos_scores1 = torch.bmm(query_embeds, torch.transpose(raw_doc_embeds_pos, 1, 2))
        neg_scores1 = torch.bmm(query_embeds, torch.transpose(raw_doc_embeds_neg, 1, 2))
        pos_scores2 = torch.bmm(raw_query_embeds, torch.transpose(doc_embeds_pos, 1, 2))
        neg_scores2 = torch.bmm(raw_query_embeds, torch.transpose(doc_embeds_neg, 1, 2))
        loss2 = torch.mean(1 - self.tanh(pos_scores1) + self.tanh(neg_scores1))
        loss3 = torch.mean(1 - self.tanh(pos_scores2) + self.tanh(neg_scores2))
        loss = loss1 + 0.1 * loss2 + 0.1 * loss3
        return loss
