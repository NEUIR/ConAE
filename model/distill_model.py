from torch import nn


class ConAE_model(nn.Module):
    def __init__(self, args):
        super(ConAE_model, self).__init__()
        self.query_encoder = nn.Linear(args.input_dim, args.output_dim, bias=True)
        self.doc_encoder = nn.Linear(args.input_dim, args.output_dim, bias=True)


    def enocde_queries(self, query_embeds):
        query_embeds = self.query_encoder(query_embeds)
        return query_embeds

    def enocde_docs(self, doc_embeds):
        doc_embeds = self.doc_encoder(doc_embeds)
        return doc_embeds

    def forward(self):
        pass

