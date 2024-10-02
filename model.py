import torch
import torch.nn as nn

# Model parameters
num_chars = 81
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


class AttentionHead(nn.Module):
    """ 
    Single attention head class:
    - takes in a sequence of embeddings and returns a sequence of the same length
    """
    def __init__(self, embedding_dim):
        super(AttentionHead, self).__init__()
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        # register a buffer to store the mask
        self.register_buffer('mask', torch.tril(torch.ones(seq_dim, seq_dim))) 

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        seq_dim = x.shape[1]

        # compute attention weights by taking the dot product of query and key
        attention_weights =  torch.matmul(q, k.transpose(-2, -1)) 
        # scale by dividing by sqrt(embedding_dim)
        attention_weights = attention_weights / torch.sqrt(torch.tensor(embedding_dim).float()) 
        # apply mask to attention weights for causality
        attention_weights = attention_weights.masked_fill(self.mask[:seq_dim,:seq_dim] == 0, float('-inf')) 
        # apply softmax to attention weights
        attention_weights = torch.softmax(attention_weights, dim=-1) 
        # apply dropout to attention weights
        attention_weights = self.dropout(attention_weights) 
        # compute output by taking the dot product of attention weights and value
        output = torch.matmul(attention_weights, v) 

        return output


class MultiAttentionHead(nn.Module):
    """ 
    Multi-head attention class:
    - takes in a sequence of embeddings and returns a sequence of the same length
    - uses multiple attention heads in parallel
    """

    def __init__(self, embedding_dim, n_heads):
        super(MultiAttentionHead, self).__init__()
        self.heads = nn.ModuleList([AttentionHead(embedding_dim) for _ in range(n_heads)])
        self.summarize = nn.Linear(embedding_dim * n_heads, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # apply each attention head to the input sequence
        head_outputs = [head(x) for head in self.heads] 
        # concatenate the outputs of each head to get a single tensor
        multihead_output = torch.cat(head_outputs, dim=-1) 
        # apply a linear layer to summarize the multi-head output to the original embedding dimension
        output = self.summarize(multihead_output)
        # apply dropout to the output 
        output = self.dropout(output) 

        return output

class FeedForward(nn.Module):
    """
    Feedforward network class:
    - applies a feedforward network to a sequence of embeddings
    - returns a sequence of the same length
    """

    def __init__(self, embedding_dim):
        super(FeedForward, self).__init__()
        self.step = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.step(x)

class DecoderBlock(nn.Module):
    """
    Decoder block class:
    - contains a multi-head attention layer and a feedforward network
    - applies layer normalization after each sublayer
    - applies a skip connection around each sublayer
    - returns a sequence of the same length
    """

    def __init__(self, embedding_dim):
        super(DecoderBlock, self).__init__()
        self.multihead = MultiAttentionHead(embedding_dim, n_heads)
        self.feedforward = FeedForward(embedding_dim)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        # apply multi-head attention
        x_multi = self.multihead(x) 
        # add the multi-head output to the input (skip connection)
        x = x + x_multi 
        # apply layer normalization
        x = self.norm1(x) 
        # apply feedforward network
        x_ff = self.feedforward(x) 
        # add the feedforward output to the input (skip connection)
        x = x + x_ff 
        # apply layer normalization
        x = self.norm2(x) 

        return x


class my_GPT(nn.Module):
    def __init__(self):
        super(my_GPT, self).__init__()
        self.token_embedding = nn.Embedding(num_chars, embedding_dim)
        self.position_embedding = nn.Embedding(seq_dim, embedding_dim)
        self.layers = nn.ModuleList([DecoderBlock(embedding_dim) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.linear = nn.Linear(embedding_dim, num_chars) # linear layer to map embeddings to character logits
        #self.softmax = nn.Softmax(dim=-1) # softmax layer to convert logits to probabilities

    def forward(self, x, y=None):
        seq_dim_x = x.shape[1]

        # Apply token embedding to input sequence
        token_emb =  self.token_embedding(x)
        # Apply position embedding to input sequence
        pos_emb = self.position_embedding(torch.arange(seq_dim_x, device=device))
        # Add token and position embeddings together to get the input embeddings for the model
        x = token_emb + pos_emb

        # Loop through the repeated transformers layers
        for layer in self.layers:
            x = layer(x)
        
        # Apply layer normalization
        x = self.layer_norm(x)
        # Apply linear layer to get logits
        y_pred = self.linear(x)

        if y is None:
            loss = None
            return y_pred, loss
        
        else:
            # Get the predicted sequence and the target sequence
            y_pred = y_pred.view(batch_size * seq_dim, num_chars) # reshape to 2D tensor for cross-entropy loss
            y = y.view(batch_size * seq_dim) # reshape to 1D tensor for cross-entropy loss
            # Compute cross-entropy loss
            loss = nn.CrossEntropyLoss()(y_pred, y)
            return y_pred, loss
    
    def generate(self, x, generation_length):
        for _ in range(generation_length):
            # Cut the input sequence to the last seq_dim tokens
            x_cut = x[:, -seq_dim:]
            # Compute the predicted sequence
            y_pred, loss = self(x_cut)
            # Get the last token of the predicted sequence
            y_pred = y_pred[:,-1,:]
            # Sample the next token from the predicted sequence
            probability = nn.Softmax(dim=-1)(y_pred)
            next_token = torch.multinomial(probability, 1)
            # Concatenate the next token to the input sequence
            x = torch.cat([x, next_token], dim=1)
        return x