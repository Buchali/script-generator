import torch
from torch import nn
from torch.nn import functional as F

from src.check_device import device
from src.params import block_size, drop_out, n_blocks, n_embd, n_heads


class Head(nn.Module):
    """
    Attention Head Module.
    """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size)
        self.query = nn.Linear(n_embd, head_size)
        self.value = nn.Linear(n_embd, head_size)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.drop_out = nn.Dropout(drop_out)

    def forward(self, X):
        B, T, C = X.shape
        k = self.key(X)
        q = self.query(X)
        wei = k @ q.transpose(-2, -1) * (C**(-0.5))
        wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        wei = self.drop_out(wei)
        v = self.value(X)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """
    Multiple attention head concatenated together.
    """
    def __init__(self, head_size, n_heads):
        super().__init__()
        self.heads = nn.ModuleList(Head(head_size) for _ in range(n_heads))
        self.proj = nn.Linear(n_embd, n_embd)
        self.drop_out = nn.Dropout(drop_out)

    def forward(self, X):
        out = torch.cat([h(X) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.drop_out(out)
        return out

class FeedForward(nn.Module):
    """
    A simple FeedForward Net.
    """
    def __init__(self, n_hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_hidden, 4*n_hidden),
            nn.ReLU(),
            nn.Linear(4*n_hidden, n_hidden),
            nn.Dropout(drop_out)
        )

    def forward(self, X):
        return self.net(X)

class Block(nn.Module):
    """
    Transformer Block containing Multiple Head Attention, Layer Norm, and FeedForward Connection.
    """
    def __init__(self, n_embd, n_heads):
        super().__init__()
        head_size = n_embd // n_heads
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.heads = MultiHeadAttention(head_size, n_heads)
        self.ff = FeedForward(n_embd)

    def forward(self, X):
        X = X + self.heads(self.ln1(X))
        X = X + self.ff(self.ln2(X))
        return X

class BigramLanguageModel(nn.Module):
    def __init__(self, char_size):
        super().__init__()
        self.tok_embd_table = nn.Embedding(num_embeddings=char_size ,embedding_dim=n_embd)
        self.pos_embd_table = nn.Embedding(num_embeddings=block_size ,embedding_dim=n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_heads).to(device) for _ in range(n_blocks)]
        )
        self.lm_head = nn.Linear(n_embd, char_size)
        self.ln_f = nn.LayerNorm(char_size)
    def forward(self, X, targets=None):
        B, T = X.shape
        tok_embd = self.tok_embd_table(X)
        pos_embd = self.pos_embd_table(torch.arange(T, device=device))
        x = tok_embd + pos_embd
        x = self.blocks(x)
        logits = self.lm_head(x)
        logits = self.ln_f(logits)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            targets = targets.view(B*T)
            logits = logits.view(B*T, C)

            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
