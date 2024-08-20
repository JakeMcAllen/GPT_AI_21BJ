from torch.nn import functional as F
from torchtext.data.utils import get_tokenizer
import torch.nn as nn
import torch

from datasets import load_dataset, load_from_disk
from transformers import DistilBertTokenizer
import transformers

from tqdm import tqdm
import pandas as pd
import csv
import pandas as pd
import os
import requests



# ------------
# SENTENCE MATRIX
batch_size = 8                 # 16                 # How many independent sequences will we process in parallel ( impact the loss calculation time )
block_size = 250               # 500                # What is the maximum context length for predictions?

# LOSS
eval_interval = 50        # 100                     # Number iteraction when start to evaluate the loss



eval_iters = 10          # 50 - 200                 # Times of testing the LOSS ( impact the loss calculation time )
learning_rate = 1e-3

# DEVICE
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
# print(device)

# BLOCKS
n_embd = 64
n_head = 32
n_layer = 32              # 4                  # Number of layers
dropout = 0.3             # 0.0

# ITERACTION
max_iters = 50000          # 100000                # Number of generations
iterator_dataset = 0
minStrLength = 256

### To modify
# eval_interval eval_iters & max_iters




# ------------
# SET VARIABLES FOR VALUE REGISTRATION:
checkpoint_save = False
savingBasePath = f"save/model_nn_{str(n_layer)}_gen_{str(max_iters)}"
model_name = f'model_nn_{str(n_layer)}_gen_{str(max_iters)}.pt'
csv_file_name = f'loss_nn_{str(n_layer)}_gen_{str(max_iters)}.csv'

fields = ['train_loss', 'step']




# ------------
# LOAD DATASETS AND TOKENIZER:
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

torch.manual_seed(1337)
vocab_size = tokenizer.vocab_size


# Loadign pre-training bert tokenizer
encode = lambda s: tokenizer.encode_plus(s, padding="max_length", return_tensors='pt', add_special_tokens=True, max_length=block_size, truncation=True)['input_ids'].flatten().tolist()
decode = lambda s: tokenizer.decode(s)


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

    def sv(self, basePath, number_of_block):
        torch.save(self.net.state_dict(), f"{basePath}/FeedFoward_n{number_of_block}.pt")

    def map(self, basePath, number_of_block, device):
        self.net.load_state_dict(torch.load(f"{basePath}/FeedFoward_n{number_of_block}.pt", map_location=torch.device(device)))


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

    def sv(self, basePath, number_of_block):
        self.ffwd.sv(basePath, number_of_block)

    def map(self, basePath, number_of_block, device):
        self.ffwd.map(basePath, number_of_block, device)
        # torch.load(f"{base_path}/blocks_BigramLanguageModel.pt", map_location=torch.device(device))


# super simple bigram model

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)                        # 64, 28

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

    def sv(self, base_path):
        torch.save(self.blocks.state_dict(), f"{base_path}/blocks_BigramLanguageModel.pt")

        for i in range(n_layer):
            self.blocks[i].sv(base_path, i)

    def map(self, device, base_path):
        for i in range(n_layer):
            self.blocks[i].map(base_path, i, device)

        self.blocks.load_state_dict(torch.load(f"{base_path}/blocks_BigramLanguageModel.pt", map_location=torch.device(device)))

"""### Utility functions"""

def reasume_base_data(path_to_save, device):
    model = BigramLanguageModel()
    model.map(device, "/".join(path_to_save.split("/")[:-1]))
    model.load_state_dict(torch.load(path_to_save))
    print("\n LOAD \n")
    return model.to(device)



#----------------------------
#   PRINT
#----------------------------
def run_model(m, device, start_sentence, max_new_tokens=50, commentFlag=False, decoded=False, eliminate_input=True):
    inptVal = encode( start_sentence )
    inptVal = inptVal[:inptVal.index(102)]
    inptVal.append(101)
    inptVal_c = inptVal
    inptVal = torch.as_tensor(inptVal, dtype=torch.long, device=device)

    outVal = m.generate(inptVal.view(1, inptVal.size()[0]), max_new_tokens=max_new_tokens)
    outVal = outVal[0].tolist()

    if eliminate_input: outVal = outVal[len(inptVal_c):]
    if decoded: outVal = decode(outVal)
    if commentFlag: print(f" Reponse to str: {start_sentence} => \nOutput predicted:\t  :{outVal} ", end="\n\n")

    return outVal


def run_model_zeros(m, device, max_new_tokens=50, commentFlag=False):
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    outVal = decode(m.generate(context, max_new_tokens=50)[0].tolist())

    if commentFlag: print(f"\nText: {outVal}")

    return outVal















"""# Tests
This version is from running model

## Load model
"""

# Variable setting


baseName = "RLHF" # "model"
savingBasePath = f"./../../../save/model_nn_{str(n_layer)}_gen_{str(max_iters)}"
model_name = f'{baseName}_model_nn_{str(n_layer)}_gen_{str(max_iters)}.pt'
csv_file_name = f'{baseName}_loss_nn_{str(n_layer)}_gen_{str(max_iters)}.csv'

print(f"\n\n{savingBasePath}/{model_name}\n\n\n")


# Model
# m = reasume_base_data(f"./{savingBasePath}/model_{model_name}", "cpu")
m = reasume_base_data(f"{savingBasePath}/{model_name}", "cpu")



# plot_loss_graph(csv_file_name=f"./{savingBasePath}/{csv_file_name}")
# print("HW: \t" + run_model(m , device, "Hello world", decoded=True), end="\n\n\n\n")
# print("ZI: \t" + run_model_zeros(m, device), end="\n\n\n\n")

"""## Test model
"""



def get_model_response(query):
    return run_model(m , device, query, decoded=True)

if __name__=="__main__":
    while True:
        context = input(">>> ")
        if context=="exit": break

        print("HW: \t" + run_model(m , device, context, decoded=True), end="\n\n\n\n")