import torch
import math
define positional_embeddings(seq_len, embed_dim):
  position = torch.arange(seq_len).unsqueeze(1)
  div_term  = torch.exp(torch.arange(0, embed_dim, 2)*-(math.log(10000.0)/embed_dim))
  pe = torch.zeros(seq_len, embed_dim)
  pe[:, 0::2] = torch.sin(position * div_term)
  pe[:, 1::2] = torch.cos(position * div_term)
  return pe.unsqueeze(0)
