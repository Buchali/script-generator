import torch

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

# check for the available device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available and being used")
else:
    device = torch.device("cpu")
    print("GPU is not available, using CPU instead")

