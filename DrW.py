import typing
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class Attention(nn.Module):
    """
    Attention module.

    :param input_size: Size of input.
    :param mask: An integer to mask the invalid values. Defaults to 0.

    Examples:
        >>> import torch
        >>> attention = Attention(input_size=10)
        >>> x = torch.randn(4, 5, 10)
        >>> x.shape
        torch.Size([4, 5, 10])
        >>> x_mask = torch.BoolTensor(4, 5)
        >>> attention(x, x_mask).shape
        torch.Size([4, 5])

    """

    def __init__(self, input_size: int = 100):
        """Attention constructor."""
        super().__init__()
        self.linear = nn.Linear(input_size, 1, bias=False)

    def forward(self, x, x_mask):
        """Perform attention on the input."""
        x = self.linear(x).squeeze(dim=-1)
        x = x.masked_fill(x_mask, -float('inf'))
        return F.softmax(x, dim=-1)



class DrW(nn.Module):
    """
    DrW Model.
    """
    
    def __init__(self, model_params):
        super(DrW, self).__init__()
        self._params = model_params

    def _make_default_embedding_layer(
        self,
        **kwargs
    ) -> nn.Module:
        """:return: an embedding module."""
        if isinstance(self._params['embedding'], np.ndarray):
            self._params['embedding_input_dim'] = (
                self._params['embedding'].shape[0]
            )
            self._params['embedding_output_dim'] = (
                self._params['embedding'].shape[1]
            )
            return nn.Embedding.from_pretrained(
                embeddings=torch.Tensor(self._params['embedding']),
                freeze=self._params['embedding_freeze'],
                padding_idx=self._params['padding_idx']
            )
        else:
            return nn.Embedding(
                num_embeddings=self._params['embedding_input_dim'],
                embedding_dim=self._params['embedding_output_dim'],
                padding_idx=self._params['padding_idx']
            )
        
    def _make_output_layer(
        self,
        in_features: int = 0
    ):
        """:return: a correctly shaped torch module for model output."""

        out_features = 1
        if self._params['out_activation_func']=='tanh':
            return nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.Tanh()
            )
        else:
            return nn.Linear(in_features, out_features)

    def _make_perceptron_layer(
        self,
        in_features: int = 0,
        out_features: int = 0,
        activation: nn.Module = nn.ReLU()
    ) -> nn.Module:
        """:return: a perceptron layer."""
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            activation
        )

    def _make_multi_layer_perceptron_layer(self, in_features) -> nn.Module:
        """:return: a multiple layer perceptron."""

        if self._params['mlp_activation_func'] == 'tanh':
            activation = nn.Tanh()
        elif self._params['mlp_activation_func'] == 'sigm':
            activation = nn.Sigmoid()
        mlp_sizes = [
            in_features,
            *self._params['mlp_num_layers'] * [self._params['mlp_num_units']],
            self._params['mlp_num_fan_out']
        ]
        mlp = [
            self._make_perceptron_layer(in_f, out_f, activation)
            for in_f, out_f in zip(mlp_sizes, mlp_sizes[1:])
        ]
        return nn.Sequential(*mlp)

    def build(self):
        """Build model structure."""
        self.embedding = self._make_default_embedding_layer()
        self.attention = Attention(
            input_size=self._params['embedding_output_dim']
        )
        self.mlp = self._make_multi_layer_perceptron_layer(
            self._params['top_k']
        )
        self.out = self._make_output_layer(1)
        
        # wlstm
        self.left_bilstm = nn.LSTM(
            input_size=self._params['embedding_output_dim'],
            hidden_size=self._params['hidden_size'],
            num_layers=self._params['num_layers'],
            batch_first=True,
            dropout=self._params['dropout_rate'],
            bidirectional=True
        ) 

        self.linear = nn.Linear(self._params['hidden_size']*2, 2)

        #for wc
        geo_emblens = 64
        self.latgps_embedding = nn.Embedding(
            num_embeddings=1000,
            embedding_dim=geo_emblens
        )
        self.longps_embedding = nn.Embedding(
            num_embeddings=1000,
            embedding_dim=geo_emblens
        )

        self.linear3 = nn.Linear(self._params['embedding_output_dim'], 10)
        self.linear5 = nn.Linear(geo_emblens, 32)
        self.linear4 = nn.Linear(164, 2)

        # self.linear4 = nn.Linear(84, 2)

        #for w
        self.linear1 = nn.Linear(self._params['embedding_output_dim'], 10)
        self.linear2 = nn.Linear(100, 2)

    
    def forward1(self, inputs):
        """Forward."""
        query, doc = inputs['text_left'], inputs['text_right']
        # shape = [B, L]
        mask_query = (query == self._params['mask_value'])
        # Process left input.
        # shape = [B, L, D]
        embed_query = self.embedding(query.long())
        # shape = [B, R, D]
        embed_doc = self.embedding(doc.long())
        # Matching histogram of top-k
        # shape = [B, L, R]
        matching_matrix = torch.einsum(
            'bld,brd->blr',
            F.normalize(embed_query, p=2, dim=-1),
            F.normalize(embed_doc, p=2, dim=-1)
        )
        # shape = [B, L, K]
        matching_topk = torch.topk(
            matching_matrix,
            k=self._params['top_k'],
            dim=-1,
            sorted=True
        )[0]
        # shape = [B, L]
        attention_probs = self.attention(embed_query, mask_query)

        # shape = [B, L]
        dense_output = self.mlp(matching_topk).squeeze(dim=-1)
        x = torch.einsum('bl,bl->b', dense_output, attention_probs).unsqueeze(dim=-1)
        x = self.out(x)
        return x

    def forward(self, inputs):
        """Forward."""
        with torch.no_grad():
            # Scalar dimensions referenced here:
            #   B = batch size (number of sequences)
            #   D = embedding size
            #   L = `input_left` sequence length
            #   R = `input_right` sequence length
            #   K = size of top-k

            # Left input and right input.
            # shape = [B, L]
            # shape = [B, R]
            query, doc = inputs['text_left'], inputs['text_right']
            # shape = [B, L]
            mask_query = (query == self._params['mask_value'])
            # Process left input.
            # shape = [B, L, D]
            embed_query = self.embedding(query.long())
            # shape = [B, R, D]
            embed_doc = self.embedding(doc.long())
            # Matching histogram of top-k
            # shape = [B, L, R]
            matching_matrix = torch.einsum(
                'bld,brd->blr',
                F.normalize(embed_query, p=2, dim=-1),
                F.normalize(embed_doc, p=2, dim=-1)
            )
            # shape = [B, L, K]
            matching_topk = torch.topk(
                matching_matrix,
                k=self._params['top_k'],
                dim=-1,
                sorted=True
            )[0]
            # shape = [B, L]
            attention_probs = self.attention(embed_query, mask_query)
            # shape = [B, L]
            dense_output = self.mlp(matching_topk).squeeze(dim=-1)
            x = torch.einsum('bl,bl->b', dense_output, attention_probs).unsqueeze(dim=-1)
            x = self.out(x)

        query_lat = inputs['location_left'][:, 0].long()
        query_lon = inputs['location_left'][:, 1].long()
        lat_embs = self.latgps_embedding(query_lat)
        lon_embs = self.longps_embedding(query_lon)
        lat_embs = self.linear5(lat_embs)
        lon_embs = self.linear5(lon_embs)

        query_l1 = torch.flatten(self.linear3(embed_query), start_dim=1)
        query_t_c = torch.cat([query_l1, lat_embs, lon_embs], dim=-1)
        w = self.linear4(query_t_c)
        # w = F.normalize(w, p=2, dim=1)
        # w = torch.softmax(w, dim=1)

        distance = inputs['distance'].float()
        distance = distance.unsqueeze(1)
        x_all = torch.cat([x, distance], dim=1)
        pred = torch.sum(w * x_all, dim=1).unsqueeze(1)

        return pred