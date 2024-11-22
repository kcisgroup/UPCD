import torch
import math
import torch.nn as nn
from torch.nn import functional as F
from my_models.residual import ResidualStack
from vector_quantize_pytorch import VectorQuantize

class Symbolization(nn.Module):
    def __init__(self, args):
        super(Symbolization, self).__init__()

        self.symbol_space = args.symbol_space
        self.fea_num = args.feature_dim
        self.h_dim = args.h_dim
        self.n_res_layers = args.n_res_layers

        self.encoder = nn.Sequential(
            nn.Linear(1, self.h_dim // 4),
            nn.LayerNorm(normalized_shape=[self.h_dim // 4], elementwise_affine=True),
            nn.Tanh(),
            nn.Linear(self.h_dim // 4, self.h_dim // 2),
            nn.LayerNorm(normalized_shape=[self.h_dim // 2], elementwise_affine=True),
            nn.Tanh(),
            nn.Linear(self.h_dim // 2, self.h_dim),
            ResidualStack(self.h_dim, self.h_dim, self.h_dim // 2, self.n_res_layers),
        )

        self.symbol_layer = torch.nn.ModuleList([
            # dim:表示输入向量的维度     codebook_dim:指定码本中每个向量的维度，理想情况下应该与dim相同   codebook_size:码本中向量的数量
            VectorQuantize(dim=self.h_dim, codebook_dim=self.h_dim, codebook_size=self.symbol_space) for _ in range(self.fea_num)
        ])

        self.decoder = nn.Sequential(
            ResidualStack(self.h_dim, self.h_dim, self.h_dim // 2, self.n_res_layers),
            nn.Linear(self.h_dim, self.h_dim // 2),
            nn.LayerNorm(normalized_shape=[self.h_dim // 2], elementwise_affine=True),
            nn.Tanh(),
            nn.Linear(self.h_dim // 2, self.h_dim // 4),
            nn.LayerNorm(normalized_shape=[self.h_dim // 4], elementwise_affine=True),
            nn.Tanh(),
            nn.Linear(self.h_dim // 4, 1),
        )

    def forward(self, x):
        batch_size, seq_len = x.shape
        # (batch, seq, 1)
        inputs = x.unsqueeze(-1)
        # (batch, seq, h_dim)
        inputs = self.encoder(inputs)
        quantized_encoded_data = torch.zeros_like(inputs)
        commit_losses = []
        one_hot_encoded_data = torch.zeros((inputs.shape[0], inputs.shape[1], self.symbol_space), dtype=torch.float32)
        for i in range(seq_len):
            current_position_vectors = inputs[:, i:i+1, :]
            symbol_layer = self.symbol_layer[i]

            quantized_current_position, indices, commit_loss = symbol_layer(current_position_vectors)
            index_one_hot = F.one_hot(indices.squeeze(1), num_classes=self.symbol_space).float()
            one_hot_encoded_data[:, i, :] = index_one_hot
            quantized_encoded_data[:, i, :] = quantized_current_position.squeeze(1)
            commit_losses.append(commit_loss)

        total_commit_loss = sum(commit_losses) / seq_len

        output = self.decoder(quantized_encoded_data)
        output = output.view([batch_size, -1])

        recons_loss = F.mse_loss(output, x)
        symbol_loss = recons_loss + total_commit_loss
        # symbol_loss = total_commit_loss
        return output, symbol_loss, one_hot_encoded_data

    def loss_function(self, output, input, commit_loss):

        recons_loss = F.mse_loss(output, input)
        loss = recons_loss + commit_loss
        return {
            'loss': loss,
            'Reconstruction_loss': recons_loss,
            'commit_loss': commit_loss
        }

if __name__ == '__main__':
    pass


