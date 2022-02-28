import math
from fast_transformers.builders import RecurrentEncoderBuilder

class RecurrentGenerator(torch.nn.Module):
    class PositionalEncoding(torch.nn.Module):
        def __init__(self, d_model, dropout=0.0, max_len=5000):
            super(RecurrentGenerator.PositionalEncoding, self).__init__()
            self.dropout = torch.nn.Dropout(p=dropout)
            self.d_model = d_model
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)

        def forward(self, x, i):
            pos_embedding =  self.pe[0, i:i+1]
            x = torch.cat(
                [x, pos_embedding.expand_as(x)],
                dim=1
            )
            return self.dropout(x)

    def __init__(self, d_model, sequence_length, mixtures,
                 attention_type="full", n_layers=4, n_heads=4,
                 d_query=32, dropout=0.1, softmax_temp=None,
                 attention_dropout=0.1):
        super(RecurrentGenerator, self).__init__()

        self.pos_embedding = self.PositionalEncoding(
            d_model//2,
            max_len=sequence_length
        )
        self.value_embedding = torch.nn.Embedding(
            256,
            d_model//2
        )
        self.transformer = RecurrentEncoderBuilder.from_kwargs(
            attention_type=attention_type,
            n_layers=n_layers,
            n_heads=n_heads,
            feed_forward_dimensions=n_heads*d_query*4,
            query_dimensions=d_query,
            value_dimensions=d_query,
            dropout=dropout,
            softmax_temp=softmax_temp,
            attention_dropout=attention_dropout
        ).get()
        self.predictor = torch.nn.Linear(
            d_model,
            mixtures * 3
        )

    def forward(self, x, i=0, memory=None):
        x = x.view(x.shape[0])
        x = self.value_embedding(x)
        x = self.pos_embedding(x, i)
        y_hat, memory = self.transformer(x, memory)
        y_hat = self.predictor(y_hat)

        return y_hat, memory