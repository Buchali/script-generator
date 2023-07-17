import torch

from src.check_device import device
from src.model import BigramLanguageModel
from src.params import block_size

# import data
with open('./src/data/Friends_Transcript.txt', 'r') as fh:
    text = fh.read()

print(f"There are a total of {len(text)} chars in FRIENDS transcript")
chars = sorted(set(text))
char_size = len(chars)
print(f'unique chars: {char_size}')
print(''.join(chars))

# tokenize
c2i = {c:i for i, c in enumerate(chars)}
i2c = {i:c for c, i in c2i.items()}

encode = lambda s: [c2i[c] for c in s]
decode = lambda l: ''.join([i2c[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

# split train and validation data
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# Learning Params
batch_size = 32
lr = 1e-3
max_iter = 20000
eval_iters = 500

# Select random batches of data for training or validation.
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data) - block_size, (batch_size, ))
    X = torch.vstack([data[i:i+block_size] for i in ix])
    y = torch.vstack([data[i+1:i+block_size+1] for i in ix])
    return X.to(device), y.to(device)

# Create Model
model = BigramLanguageModel(char_size=char_size)
model.to(device)

# Adam optimiser
opt = torch.optim.Adam(model.parameters(), lr=lr)

# A function to estimate the loss of the model.
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

torch.manual_seed(42)

# Training
for step in range(max_iter):
    X_train, y_train = get_batch('train')
    logits, loss = model(X_train, y_train)

    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    if step % eval_iters == 0:
        losses = estimate_loss()
        print(f"step {step}: train loss {losses['train']:.3f}, val loss {losses['val']:.3f}")

idx_start = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(idx_start, 1000)[0].tolist()))
