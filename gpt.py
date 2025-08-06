import torch
import torch.nn as nn
import torch.nn.functional as F
import deepspeed
import math

def positional_embeddings(seq_len, embed_dim):
  position = torch.arange(seq_len).unsqueeze(1)
  div_term  = torch.exp(torch.arange(0, embed_dim, 2)*-(math.log(10000.0)/embed_dim))
  pe = torch.zeros(seq_len, embed_dim)
  pe[:, 0::2] = torch.sin(position * div_term)
  pe[:, 1::2] = torch.cos(position * div_term)
  return pe.unsqueeze(0)

class Multihead_Attention(nn.Module):
  def __init__(self, num_heads, embed_dim):
    assert embed_dim % num_heads == 0
    self.num_heads = num_heads
    self.embed_dim = embed_dim
    self.head_dim  = embed_dim/num_heads

    self.q_proj = nn.Linear(embed_dim, embed_dim)
    self.k_proj = nn.Linear(embed_dim, embed_dim)
    self.v_proj = nn.Linear(embed_dim, embed_dim)

  def forward(self, x, mask=None):
    B, T, E = x.size()
    H, D = self.num_heads, self.head_dim

    q_proj = self.q_proj(x)
    k_proj = self.k_proj(x)
    v_proj = self.v_proj(x)
    
    q = q_proj.view(B, T, H, D).transpose(1,2)
    k = k_proj.view(B, T, H, D).transpose(1,2)
    v = v_proj.view(B, T, H, D).transpose(1,2)

    scores = torch.matmul(q, k.transpose(-2,-1))/math.sqrt(D)
    atten_scores = F.softmax(scores, dim = -1)
    atten_scores =  torch.matmul(scores, v)
    atten_output = atten_scores.transpose(1,2).contiguous().view(B, T, E)
    return atten_output

class Transformer_Block(nn.Module):
  def __init__(self, embed_dim, num_heads, ffn_hidden_dim):
    self.atten = Multihead_Attention(emed_dim, num_heads)
    self.ln1 = nn.LayerNorm(embed_dim)
    self.ln2 = nn.LayerNorm(embed_dim)
    self.ffn = nn.Sequential(
      nn.Linear(embed_dim, ffn_hidden_dim),
      nn.GeLU(),
      nn.Linear(ffn_hidden_dim, embed_dim)
    )
  def forward(self, x, mask=None):
    atten = self.atten(self.ln1(x), mask)
    x = x+atten
    ffn =  self.ffn(self.ln2(x))
    x = x+ffn
    return x


if __name__ = '__main__':
  device = ("cuda" if torch.cuda.is_available() else "cpu")
  model = gptmodel()
  criterion = nn.CrossEntropyLoss()
  model_engine, optimiser, _, _ = deepspeed.intialise(
    model = model,
    model_parameters = model.parameters()
    config = 'ds_config.json')
  
  epochs = 20
  total_loss = 0
  model.train()
  for _ in range(epochs):
    for (inputs, labels) in enumerate(train_loader):
      inputs, labels = inputs.to(device), labels.to(device)
      outputs = model_engine(inputs)
      loss = critertion(labels, outputs)
      model_engine.backward()
      model_engine.step()
      total_loss+=loss.item()
    # evaluation loop
  model_engine.eval()
  correct =  predicted = 0
  for (inputs, labels) in test_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model_engine(inputs)
    correct +=  
      


    
