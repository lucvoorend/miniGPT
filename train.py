import torch
import torch.nn as nn
from model import my_GPT

#Load the txt file as a big string
input_file = "acda_en_de_munnik_lyrics.txt"
training = True
generating = True


with open(input_file) as f:
    data = f.read()

unique_chars = sorted(list(set(data))) # get all unique characters in the text file
num_chars = len(unique_chars)
embedding_dim = 64
seq_dim = 256
batch_size = 64
n_layers = 6
n_heads = 6
generation_length = 10000
n_epochs = 5000
eval_iters = 100
lr = 3e-4
dropout = 0.1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Device:', device)

# Create a dictionary that maps each character to a unique integer (encoder and decoder)
char_to_int = {c: i for i, c in enumerate(unique_chars)}
int_to_char = {i: c for i, c in enumerate(unique_chars)}

encode = lambda string: [char_to_int[char] for char in string] 
# encoder function that takes a string and returns a list of integers
decode = lambda int_list: ''.join([int_to_char[i] for i in int_list]) 
# decoder function that takes a list of integers and returns a string

train_val_split = 0.9
train_data = encode(data[:int(len(data)*train_val_split)])
val_data = encode(data[int(len(data)*train_val_split):])

# Create batches of sequences of a fixed length as dataloader
def create_batch(data, seq_length, batch_size):
    data = torch.tensor(data)
    sample_idx = torch.randint(0, len(data) - seq_length, (batch_size,)) # randomly sample starting indices (one for each sequence)

    x = torch.zeros((batch_size, seq_length), dtype=torch.long) # initialize input tensor
    y = torch.zeros((batch_size, seq_length), dtype=torch.long) # initialize target tensor

    for i in range(batch_size):
        x[i] = data[sample_idx[i]:sample_idx[i]+seq_length]
        y[i] = data[sample_idx[i]+1:sample_idx[i]+seq_length+1]
    
    x = x.to(device)
    y = y.to(device)
    return x, y

@torch.no_grad()
def get_val_loss():
    model.eval()
    losses = torch.zeros(eval_iters)
    for i in range(eval_iters):
        x, y = create_batch(val_data, seq_dim, batch_size)
        y_pred, loss = model.forward(x,y)
        losses[i] = loss
    model.train()
    return losses.mean()

model = my_GPT()
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

if training:
    print('Training...')
    for epoch in range(n_epochs):
        model.train()
        x, y = create_batch(train_data, seq_dim, batch_size)
        y_pred, loss = model.forward(x, y)

        if epoch % 10 == 0:
            val_loss = get_val_loss()
            print(f'Epoch {epoch}, Loss: {loss.item()}, Val Loss: {val_loss.item()}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Training finished')

# Generate text
if generating:
    print('Generating text...')
    model.eval()
    start_context = torch.zeros((1,1), dtype=torch.long)
    generated_text = model.generate(start_context, generation_length)

    # Write the generated text to a file
    with open('generated_text.txt', 'w') as f:
        generated_text = decode(generated_text[0].tolist())
        f.write(generated_text)

    print('Generated text saved to generated_text.txt')
