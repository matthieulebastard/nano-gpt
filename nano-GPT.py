
"""
Code developed during the "Machine Learning" tutorial of neetcode.io,
from module 10 onwards
Correction and demo notebook is available at https://colab.research.google.com/drive/1L92UwfFlVlog-p8PhKe-ioROAaBa4aFc?usp=sharing#scrollTo=qpwXg0YitK0w
"""

import torch
from typing import List, Tuple
import torch.nn as nn

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def batch_loader(raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]]]:
    """
        inputs : 
            raw_dataset : a bunch of text from which we'll generate a list of learning sets (string)
            context_length : number of words kept in context (int)
            batch_size : size of the sets
    
        returns: 
            X : input dataset
            Y : answers (batch_size * context_length)

        Ex : 
        Input:
            raw_dataset = "Hello darkness my old friend"
            context_length = 3
            batch_size = 2

        Output:
            X = [['darkness', 'my', 'old'], ['hello', 'darkness', 'my']]
            Y = [['my', 'old', 'friend'], ['darkness', 'my', 'old']]
    """

    words_list = raw_dataset.split()
    batch_start_indices = torch.randint(low=0, high=len(words_list) - context_length,size=(batch_size,))

    # Return X : input dataset and Y : answsers
    # dim : batch_size * context_length dataset & its label
    X = []
    Y = []
    for index in batch_start_indices:
        X.append(words_list[index:index+context_length])
        Y.append(words_list[index+1:index+context_length+1])
    return X, Y


class GPT(nn.Module):
    """
        The GPT class, correction of the tutorial
        Kept here because it is compatible with the pre-trained dataset "weights.pt"
    """
    class TransformerBlock(nn.Module):

        class MultiHeadedSelfAttention(nn.Module):

            class SingleHeadAttention(nn.Module):
                def __init__(self, model_dim: int, head_size: int):
                    super().__init__()
                    self.key_layer = nn.Linear(model_dim, head_size, bias=False)
                    self.query_layer = nn.Linear(model_dim, head_size, bias=False)
                    self.value_layer = nn.Linear(model_dim, head_size, bias=False)

                def forward(self, embedded):
                    k = self.key_layer(embedded)
                    q = self.query_layer(embedded)
                    v = self.value_layer(embedded)

                    scores = q @ torch.transpose(k, 1, 2) # @ is the same as torch.matmul()
                    context_length, attention_dim = k.shape[1], k.shape[2]
                    scores = scores / (attention_dim ** 0.5)

                    lower_triangular = torch.tril(torch.ones(context_length, context_length))
                    mask = (lower_triangular == 0).to(device)
                    scores = scores.masked_fill(mask, float('-inf'))
                    scores = nn.functional.softmax(scores, dim = 2)

                    return scores @ v

            def __init__(self, model_dim: int, num_heads: int):
                super().__init__()
                self.attention_heads = nn.ModuleList()
                for i in range(num_heads):
                    self.attention_heads.append(self.SingleHeadAttention(model_dim, model_dim // num_heads))
                self.compute = nn.Linear(model_dim, model_dim)
                self.dropout = nn.Dropout(0.2)

            def forward(self, embedded):
                head_outputs = []
                for head in self.attention_heads:
                    head_outputs.append(head(embedded))
                concatenated = torch.cat(head_outputs, dim = 2)
                return self.dropout(self.compute(concatenated))

        class VanillaNeuralNetwork(nn.Module):

            def __init__(self, model_dim: int):
                super().__init__()
                self.first_linear_layer = nn.Linear(model_dim, model_dim * 4)
                self.relu = nn.ReLU()
                self.second_linear_layer = nn.Linear(model_dim * 4, model_dim)
                self.dropout = nn.Dropout(0.2) # using p = 0.2

            def forward(self, x):
                return self.dropout(self.second_linear_layer(self.relu(self.first_linear_layer(x))))

        def __init__(self, model_dim: int, num_heads: int):
            super().__init__()
            self.mhsa = self.MultiHeadedSelfAttention(model_dim, num_heads)
            self.vanilla_nn = self.VanillaNeuralNetwork(model_dim)
            self.layer_norm_one = nn.LayerNorm(model_dim)
            self.layer_norm_two = nn.LayerNorm(model_dim)

        def forward(self, embedded):
            embedded = embedded + self.mhsa(self.layer_norm_one(embedded)) # skip connection
            embedded = embedded + self.vanilla_nn(self.layer_norm_two(embedded)) # another skip connection
            return embedded

    def __init__(self, vocab_size: int, context_length: int, model_dim: int, num_blocks: int, num_heads: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, model_dim)
        self.pos_embedding = nn.Embedding(context_length, model_dim)
        self.transformer_blocks = nn.Sequential()
        for i in range(num_blocks):
            self.transformer_blocks.append(self.TransformerBlock(model_dim, num_heads))
        self.layer_norm_three = nn.LayerNorm(model_dim)
        self.vocab_projection = nn.Linear(model_dim, vocab_size)

    def forward(self, context):
        embedded = self.token_embedding(context)
        context_length = context.shape[1]
        positions = torch.arange(context_length).to(device)
        embedded = embedded + self.pos_embedding(positions)

        raw_output = self.vocab_projection(self.layer_norm_three(self.transformer_blocks(embedded)))
        # raw_output is batch by context_length by vocab_size

        return raw_output
    


class MyGPT(nn.Module):
    """
        Code developed during the tutorial. Not compatible with pre-trained model, due to diffrent variable names
        Next step will be to code the training functions, so I can make it work with any input dataset :)

    """

    def __init__(self, vocab_size: int, context_length: int, model_dim: int, num_blocks: int, num_heads: int):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, model_dim)
        self.positional_encoding = nn.Embedding(context_length, model_dim)
        self.transformer_blocks = nn.Sequential()
        for i in range(num_blocks):
            self.transformer_blocks.append(self.TransformerBlock(model_dim, num_heads))
        self.final_layer_norm = nn.LayerNorm(model_dim)
        self.final_linear = nn.Linear(model_dim,vocab_size)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, context):
    #   def forward(self, context: TensorType[int]) -> TensorType[float]:
        embedded = self.word_embedding.forward(context)
        context_length = context.shape[1]
        # positions = torch.arange(context_length)
        positions = torch.arange(context_length).to(device)
        embedded = embedded + self.positional_encoding(positions)
        transformer_output = self.transformer_blocks.forward(embedded)
        output = self.final_layer_norm(transformer_output)
        output = self.final_linear(output)
        output = self.softmax (output)
        # output = nn.functional.softmax(self.final_linear(self.final_layer_norm(transformer_output)), dim=-1)
        return torch.round(output, decimals = 4)

    class TransformerBlock(nn.Module):
        
        def __init__(self, model_dim: int, num_heads: int):
            super().__init__()
            self.layer_norm_mmha = nn.LayerNorm(model_dim)
            self.mmha_gen = self.MultiHeadedSelfAttention(model_dim, num_heads)
            self.layer_norm_ff = nn.LayerNorm(model_dim)
            self.ff_gen = self.VanillaNeuralNetwork(model_dim)

        def forward(self, embedded):
        #    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
            # Round answer to 4 decimal places
            mmha_res = self.mmha_gen.forward(self.layer_norm_mmha(embedded)) + embedded
            ff_res = self.ff_gen.forward(self.layer_norm_ff(mmha_res)) + mmha_res
            return torch.round(ff_res, decimals=4)

        class MultiHeadedSelfAttention(nn.Module):
            
            def __init__(self, model_dim: int, num_heads: int):
            # def __init__(self, embedding_dim: int, attention_dim: int, num_heads: int):
                super().__init__()
            
                # Use self.SingleHeadAttention(embedding_dim, head_size) to instantiate. You have to calculate head_size.
                head_size = model_dim // num_heads
                self.attention_heads = nn.ModuleList([self.SingleHeadAttention(model_dim, head_size) for i in range(num_heads)])

            def forward(self, embedded):
                # def forward(self, embedded: TensorType[float]) -> TensorType[float]:
                outputs = []
                for head in self.attention_heads:
                    outputs.append(head.forward(embedded))
                output_cat = torch.cat(outputs, dim=2)
                return torch.round(output_cat, decimals=4)

            class SingleHeadAttention(nn.Module):
                
                def __init__(self, embedding_dim: int, attention_dim: int):
                    super().__init__()
                    self.key_layer = nn.Linear(embedding_dim, attention_dim, bias=False)
                    self.query_layer = nn.Linear(embedding_dim, attention_dim, bias=False)
                    self.value_layer = nn.Linear(embedding_dim, attention_dim, bias=False)
                
                def forward(self, embedded):
                    # def forward(self, embedded: TensorType[float]) -> TensorType[float]:
                    key = self.key_layer(embedded)
                    query = self.query_layer(embedded)
                    values = self.value_layer(embedded)
                    raw_scores = torch.matmul(query,torch.transpose(key,1,2))
                    
                    context_length, attention_dim = key.shape[1], key.shape[2]

                    scores = raw_scores / (attention_dim ** 0.5)

                    lower_triangular_ones = torch.tril(torch.ones(context_length, context_length))
                    upper_triangular_mask = (lower_triangular_ones == 0)
                    scores = scores.masked_fill(upper_triangular_mask, float('-inf'))
                    scores = nn.functional.softmax(scores, dim=2)
                    output = torch.round(torch.matmul(scores,values), decimals=4)
                    return output
                
        class VanillaNeuralNetwork(nn.Module):

            def __init__(self, model_dim: int):
                super().__init__()
                self.up_projection = nn.Linear(model_dim, model_dim * 4)
                self.relu = nn.ReLU()
                self.down_projection = nn.Linear(model_dim * 4, model_dim)
                self.dropout = nn.Dropout(0.2) # using p = 0.2
            
            def forward(self, x):
            #   def forward(self, x: TensorType[float]) -> TensorType[float]:
                return self.dropout(self.down_projection(self.relu(self.up_projection(x))))


def generate(model, new_chars: int, context, context_length: int, int_to_char: dict):
    """
        Wrapper function to execute the model char by char. Copied from the Notebook correction.
    """
    result = []
    for i in range(new_chars):

        # adding to context and cropping to max size if necessary
        if context.shape[1] > context_length:
            context = context[:,- context_length]

        # calling the model
        vocab_probabilities = model.forward(context)
        last_time_step = vocab_probabilities[:,-1,:]
        vocab_probabilities = nn.functional.softmax(last_time_step, dim=-1)

        # Chosing a likely output among the vocabulary probabilities
        output_ints = torch.multinomial(vocab_probabilities, 1)
        # generator.set_state(initial_state)
        output_chars = output_ints          
        # adding to context and cropping to max size if necessary
        context = torch.cat((context,output_chars), dim=-1)
        # converting the output to chars
        result.append(int_to_char[output_chars.item()])
    
    return ''.join(result)


if __name__ == '__main__':
    # !git clone https://github.com/gptandchill/drake-lyric-generator
    # cd drake-lyric-generator

    vocab_size = 104
    context_length = 128
    model_dim = 252
    num_blocks = 6
    num_heads = 6

    model = GPT(vocab_size, context_length, model_dim, num_blocks, num_heads).to(device)
    WEIGHT_PATH = 'weights.pt' # Adjust as necessary
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device, weights_only=True))
    model.eval()
    new_chars = 128 # BUG: "IndexError: tuple index out of range" when new_chars > 128 (model_dim ?)
    context = torch.zeros(1, 1, dtype = torch.int64).to(device)

    int_to_char = {0: '\n', 1: ' ', 2: '!', 3: '"', 4: '$', 5: '%', 6: '&', 7: "'", 8: '(', 9: ')', 10: '*', 11: '+', 12: ',', 13: '-', 14: '.', 15: '/', 16: '0', 17: '1', 18: '2', 19: '3', 20: '4', 21: '5', 22: '6', 23: '7', 24: '8', 25: '9', 26: ':', 27: ';', 28: '?', 29: 'A', 30: 'B', 31: 'C', 32: 'D', 33: 'E', 34: 'F', 35: 'G', 36: 'H', 37: 'I', 38: 'J', 39: 'K', 40: 'L', 41: 'M', 42: 'N', 43: 'O', 44: 'P', 45: 'Q', 46: 'R', 47: 'S', 48: 'T', 49: 'U', 50: 'V', 51: 'W', 52: 'X', 53: 'Y', 54: 'Z', 55: '[', 56: ']', 57: '_', 58: 'a', 59: 'b', 60: 'c', 61: 'd', 62: 'e', 63: 'f', 64: 'g', 65: 'h', 66: 'i', 67: 'j', 68: 'k', 69: 'l', 70: 'm', 71: 'n', 72: 'o', 73: 'p', 74: 'q', 75: 'r', 76: 's', 77: 't', 78: 'u', 79: 'v', 80: 'w', 81: 'x', 82: 'y', 83: 'z', 84: '{', 85: '|', 86: '}', 87: 'à', 88: 'á', 89: 'è', 90: 'é', 91: 'ë', 92: 'ñ', 93: 'ó', 94: 'ú', 95: '\u2005', 96: '–', 97: '—', 98: '‘', 99: '’', 100: '“', 101: '”', 102: '…', 103: '\u205f'}

    print(generate(model, new_chars,context,
               context_length,
               int_to_char))