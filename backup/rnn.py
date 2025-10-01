"""
python rnn.py -d cuda -l rnn_100_E32_H256_lr0.01_N50k_b64_seed42 -iseq 50 -tseq 50 -p 3 -oseq 50 -toseq 50 -b 64 -em 16 -hs 256 -c rnn -ep 10 -lr 0.005 -s 42 -trs 10000 -ts 1000
"""

import torch
import torch.nn as nn
import numpy as np
import time
import pickle as pkl
from tqdm import tqdm
from loguru import logger
import logging
import matplotlib.pyplot as plt
import torch.nn.functional as F
import argparse
import os

class RNNCell(torch.nn.Module):
    """A single step RNN in pytorch

    h_t = tanh(x_t * W_x + h_{t-1} * W_h + b)
    """
    def __init__(self, input_size, hidden_size):
        """for a single step

        Args:
            input_size (float): [batch_size, input_size]
            hidden_size (float): [batch_size, hidden_size]
        """
        super(RNNCell, self).__init__()
        self.hidden_size = hidden_size
        self.W_x = torch.nn.Parameter(torch.randn(input_size,
                                                  hidden_size) * 0.1)
        self.W_h = torch.nn.Parameter(torch.randn(hidden_size,
                                                  hidden_size) * 0.1)
        self.b_h = torch.nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x_t, h_prev):
        """forward pass from input to hidden state

        Args:
            x_t (float): [batch_size, input_size]
            h_prev (float): [batch_size, hidden_size]
        """
        h_t = torch.tanh(x_t @ self.W_x + h_prev @ self.W_h + self.b_h)
        return h_t
    
class GRUCell(torch.nn.Module):
    """A single pass through a GRU cell

    z_t = tanh(x_t * W_z + h_{t-1} * U_z + b_z)
    r_t = tanh(x_t * W_r + h_{t-1} * U_r + b_r)
    h~_t = tanh(x_t * W_h + (r_t dot h_{t-1}) * U_h + b_h)
    h_t = (1 - z_t) dot h_{t-1} + z_t dot h~_t
    """
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()
        self.hidden_size = hidden_size
        self.W_z = torch.nn.Parameter(torch.randn(input_size, 
                                                  hidden_size) * 0.1)
        self.W_r = torch.nn.Parameter(torch.randn(input_size,
                                                  hidden_size) * 0.1)
        self.W_h = torch.nn.Parameter(torch.randn(input_size,
                                                  hidden_size) * 0.1)
        self.U_z = torch.nn.Parameter(torch.randn(hidden_size,
                                                  hidden_size) * 0.1)
        self.U_r = torch.nn.Parameter(torch.randn(hidden_size,
                                                  hidden_size) * 0.1)
        self.U_h = torch.nn.Parameter(torch.randn(hidden_size,
                                                  hidden_size) * 0.1)
        self.b_z = torch.nn.Parameter(torch.zeros(hidden_size))
        self.b_r = torch.nn.Parameter(torch.zeros(hidden_size))
        self.b_h = torch.nn.Parameter(torch.zeros(hidden_size))
        

    def forward(self, x_t, h_prev):
        """forward pass through GRU

        Args:
            x_t (float): [batch_size, input_size]
            h_prev (float): [batch_size, hidden_size]
        """
        z_t = torch.sigmoid(x_t @ self.W_z + h_prev @ self.U_z + self.b_z)
        r_t = torch.sigmoid(x_t @ self.W_r + h_prev @ self.U_r + self.b_r)
        h_tilda_t = torch.tanh(x_t @ self.W_h + (r_t * h_prev) @ self.U_h + self.b_h)
        h_t = (1 - z_t) * h_prev + z_t * h_tilda_t
        return h_t

class LSTMCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        """Initialize an LSTM cell

        Args:
            input_size (int): The input size (number of features)
            hidden_size (int): The hidden state size (number of features)
        """
        super(LSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.W_f = torch.nn.Parameter(torch.randn(input_size,
                                                  hidden_size) * 0.1)
        self.W_i = torch.nn.Parameter(torch.randn(input_size,
                                                  hidden_size) * 0.1)
        self.W_c = torch.nn.Parameter(torch.randn(input_size,
                                                  hidden_size) * 0.1)
        self.W_o = torch.nn.Parameter(torch.randn(input_size,
                                                  hidden_size) * 0.1)
        self.U_f = torch.nn.Parameter(torch.randn(hidden_size,
                                                  hidden_size) * 0.1)
        self.U_i = torch.nn.Parameter(torch.randn(hidden_size,
                                                  hidden_size) * 0.1)
        self.U_c = torch.nn.Parameter(torch.randn(hidden_size,
                                                  hidden_size) * 0.1)
        self.U_o = torch.nn.Parameter(torch.randn(hidden_size,
                                                  hidden_size) * 0.1)
        self.b_f = torch.nn.Parameter(torch.zeros(hidden_size))
        self.b_i = torch.nn.Parameter(torch.zeros(hidden_size))
        self.b_c = torch.nn.Parameter(torch.zeros(hidden_size))
        self.b_o = torch.nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x_t, h_prev, c_prev):
        """Forward pass through an LSTM cell.

        Args:
            x_t (float): [batch_size, input_size]
            h_prev (float): [batch_size, hidden_size]
            c_prev (float): [batch_size, hidden_size]

        Returns:
            h_t (float): [batch_size, hidden_size]
            c_t (float): [batch_size, hidden_size]
        """
        f_t = torch.sigmoid(x_t @ self.W_f + h_prev @ self.U_f + self.b_f)
        i_t = torch.sigmoid(x_t @ self.W_i + h_prev @ self.U_i + self.b_i)
        c_tilda_t = torch.tanh(x_t @ self.W_c + h_prev @ self.U_c + self.b_c)
        c_t = f_t * c_prev + i_t * c_tilda_t
        o_t = torch.sigmoid(x_t @ self.W_o + h_prev @ self.U_o + self.b_o)
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t
    
class mLSTMCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        Initialize an mLSTM cell.

        Args:
            input_size (int): The size of the input features.
            hidden_size (int): The size of the hidden state.

        Attributes:
            W_i (torch.nn.Parameter): Weight matrix for the input gate.
            W_m (torch.nn.Parameter): Weight matrix for modifying the input.
            U_i (torch.nn.Parameter): Recurrent weight matrix for the input gate.
            U_m (torch.nn.Parameter): Recurrent weight matrix for modifying the input.
            W_f (torch.nn.Parameter): Weight matrix for the forget gate.
            U_f (torch.nn.Parameter): Recurrent weight matrix for the forget gate.
            W_o (torch.nn.Parameter): Weight matrix for the output gate.
            U_o (torch.nn.Parameter): Recurrent weight matrix for the output gate.
            W_c (torch.nn.Parameter): Weight matrix for the cell gate.
            U_c (torch.nn.Parameter): Recurrent weight matrix for the cell gate.
            b_m (torch.nn.Parameter): Bias for modifying the input.
            b_i (torch.nn.Parameter): Bias for the input gate.
            b_o (torch.nn.Parameter): Bias for the output gate.
            b_f (torch.nn.Parameter): Bias for the forget gate.
            b_c (torch.nn.Parameter): Bias for the cell gate.
        """

        super(mLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.W_i = torch.nn.Parameter(torch.randn(input_size,
                                                  hidden_size) * 0.1)
        self.W_m = torch.nn.Parameter(torch.randn(input_size,
                                                  input_size) * 0.1)
        self.U_i = torch.nn.Parameter(torch.randn(hidden_size, 
                                                  hidden_size) * 0.1)
        self.U_m = torch.nn.Parameter(torch.randn(hidden_size,
                                                  input_size) * 0.1)
        self.W_f = torch.nn.Parameter(torch.randn(input_size,
                                                  hidden_size) * 0.1)
        self.U_f = torch.nn.Parameter(torch.randn(hidden_size,
                                                  hidden_size) * 0.1)
        self.W_o = torch.nn.Parameter(torch.randn(input_size,
                                                  hidden_size) * 0.1)
        self.U_o = torch.nn.Parameter(torch.randn(hidden_size,
                                                  hidden_size) * 0.1)
        self.W_c = torch.nn.Parameter(torch.randn(input_size, 
                                                  hidden_size) * 0.1)
        self.U_c = torch.nn.Parameter(torch.randn(hidden_size,
                                                  hidden_size) * 0.1)

        self.b_m = torch.nn.Parameter(torch.zeros(input_size))
        self.b_i = torch.nn.Parameter(torch.zeros(hidden_size))
        self.b_o = torch.nn.Parameter(torch.zeros(hidden_size))
        self.b_f = torch.nn.Parameter(torch.zeros(hidden_size))
        self.b_c = torch.nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x_t, h_prev, c_prev):
        """
        Forward pass through an mLSTM cell.

        Args:
            x_t (torch.Tensor): The input tensor at time t with shape [batch_size, input_size].
            h_prev (torch.Tensor): The previous hidden state tensor with shape [batch_size, hidden_size].
            c_prev (torch.Tensor): The previous cell state tensor with shape [batch_size, hidden_size].

        Returns:
            h_t (torch.Tensor): The next hidden state tensor with shape [batch_size, hidden_size].
            c_t (torch.Tensor): The next cell state tensor with shape [batch_size, hidden_size].
        """

        m_t = x_t @ self.W_m + h_prev @ self.U_m + self.b_m
        x_tilde_t = m_t * x_t
        i_t = torch.sigmoid(x_tilde_t @ self.W_i + h_prev @ self.U_i + self.b_i)
        f_t = torch.sigmoid(x_tilde_t @ self.W_f + h_prev @ self.U_f + self.b_f)
        o_t = torch.sigmoid(x_tilde_t @ self.W_o + h_prev @ self.U_o + self.b_o)
        c_tilde_t = torch.tanh(x_tilde_t @ self.W_c + h_prev @ self.U_c + self.b_c)
        c_t = f_t * c_prev + i_t * c_tilde_t
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t
    
class mGRU(torch.nn.Module):
    """
    source: https://arxiv.org/pdf/1907.00455
    """
    def __init__(self, input_size, hidden_size):
        """
        Initialize an mGRU cell.

        Args:
            input_size (int): The size of the input feature vector.
            hidden_size (int): The size of the hidden state vector.

        This initialization sets up the parameters for the mGRU cell, including the weight matrices
        (W_m, W_z, W_r, W_h, U_m, U_z, U_r, U_h) and bias vectors (b_m, b_z, b_r, b_h). The weight
        matrices are initialized with random values scaled by 0.1, and the bias vectors are initialized
        to zeros.
        """

        super(mGRU, self).__init__()
        self.hidden_size = hidden_size
        self.W_m = torch.nn.Parameter(torch.randn(input_size,
                                                  input_size) * 0.1)
        self.W_z = torch.nn.Parameter(torch.randn(input_size,
                                                  hidden_size) * 0.1)
        self.W_r = torch.nn.Parameter(torch.randn(input_size,
                                                  hidden_size) * 0.1)
        self.W_h = torch.nn.Parameter(torch.randn(input_size,
                                                  hidden_size) * 0.1)
        self.U_m = torch.nn.Parameter(torch.randn(hidden_size, 
                                                  input_size) * 0.1)
        self.U_z = torch.nn.Parameter(torch.randn(hidden_size, 
                                                  hidden_size) * 0.1)
        self.U_r = torch.nn.Parameter(torch.randn(hidden_size, 
                                                  hidden_size) * 0.1)
        self.U_h = torch.nn.Parameter(torch.randn(hidden_size, 
                                                  hidden_size) * 0.1)
        
        self.b_m = torch.nn.Parameter(torch.zeros(input_size))
        self.b_z = torch.nn.Parameter(torch.zeros(hidden_size))
        self.b_r = torch.nn.Parameter(torch.zeros(hidden_size))
        self.b_h = torch.nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x_t, h_prev):
        """
        Compute the next hidden state given the current input and previous hidden state.

        Args:
            x_t (torch.Tensor): The input tensor at time t with shape [batch_size, input_size].
            h_prev (torch.Tensor): The previous hidden state tensor with shape [batch_size, hidden_size].

        Returns:
            h_t (torch.Tensor): The next hidden state tensor with shape [batch_size, hidden_size].
        """
        m_t = x_t @ self.W_m + h_prev @ self.U_m + self.b_m
        x_tilde_t = m_t * x_t
        z_t = torch.sigmoid(x_tilde_t @ self.W_z + h_prev @ self.U_z + self.b_z)
        r_t = torch.sigmoid(x_tilde_t @ self.W_r + h_prev @ self.U_r + self.b_r)
        h_tilde_t = torch.tanh(x_tilde_t @ self.W_h + (r_t * h_prev) @ self.U_h + self.b_h)
        h_t = (1 - z_t) * h_prev + z_t * h_tilde_t
        return h_t

class RNN(torch.nn.Module):
    """RNN network
    """
    def __init__(self, vocab_size, embedding_size, hidden_size, 
                 device, model_choice='rnn'): # 
        """
        Initialize the RNN model.

        Args:
            vocab_size (int): The size of the vocabulary.
            embedding_size (int): The size of the embedding vectors.
            hidden_size (int): The size of the hidden state vectors.
            device (torch.device): The device on which to run the model ('cpu' or 'cuda').
            model_choice (str, optional): The type of RNN cell to use. Defaults to 'rnn'.
                Options include:
                - 'rnn': Vanilla RNN cell.
                - 'gru': GRU cell.
                - 'lstm': LSTM cell.
                - 'mlstm': Modified LSTM cell.
                - 'mgru': Modified GRU cell.

        Attributes:
            device (torch.device): The device on which the model is running.
            embedding_layer (nn.Embedding): The embedding layer for input sequences.
            hidden_size (int): The size of the hidden state vectors.
            cell (nn.Module): The RNN cell used for processing sequences.
            out_layer (nn.Linear): The linear layer for projecting hidden states to vocabulary size.
            softmax (nn.Softmax): The softmax layer for output probabilities.
        """

        super(RNN, self).__init__()

        self.device = device

        self.embedding_layer = nn.Embedding(vocab_size, 
                                            embedding_size).to(device)

        self.hidden_size = hidden_size
        # self.cell = None
        if model_choice == 'gru':
            self.cell = GRUCell(embedding_size, hidden_size)
        elif model_choice == 'lstm':
            self.cell = LSTMCell(embedding_size, hidden_size)
        elif model_choice == 'mlstm':
            self.cell = mLSTMCell(embedding_size, hidden_size)
        elif model_choice == 'mgru':
            self.cell = mGRU(embedding_size, hidden_size)
        else:
            self.cell = RNNCell(embedding_size, hidden_size)
        self.cell.to(device)
        
        # output dimension of W_out should be equal to input size for copy task
        # self.W_out = torch.nn.Parameter(torch.randn(hidden_size, vocab_size)).to(device)
        # self.b_out = torch.nn.Parameter(torch.zeros(vocab_size)).to(device)
        self.out_layer = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.Softmax(dim=1)
        # self.cell.to(device)
        self.to(device)

    # def parameters(self):
    #     params = list(super(RNN, self).parameters())
    #     if self.cell is not None:
    #         params.extend(list(self.cell.parameters()))
    #     for param in params:
    #         print(param.shape)
    #     return params

    def forward(self, X, h=None, cell_state=None, return_states=False):
        """forward pass through the RNN

        Args:
            X (float): [batch_size, seq_len, input_size]
            cell_choice (str): rnn/gru
        """
        X = self.embedding_layer(X)
        batch_size, seq_len, _ = X.shape
        # initial hidden state
        if h is None:
            h = torch.zeros(batch_size, self.hidden_size, device=X.device)
        else:
            h = h.to(X.device)
        if cell_state is not None:
            cell_state = cell_state.to(X.device)
        else:
            if isinstance(self.cell, LSTMCell) or isinstance(self.cell, mLSTMCell):
                cell_state = torch.zeros(batch_size, self.hidden_size, device=X.device)
        outputs = []
        for t in range(seq_len):
            x_t = X[:, t, :]    # [batch_size, input_size]
            if cell_state is not None:
                h, cell_state = self.cell(x_t, h, cell_state)   # [batch_size, hidden_size]
            else:
                h = self.cell(x_t, h)
            
            # Project hidden -> input_size
            # out_t = h @ self.W_out + self.b_out
            logits = self.out_layer(h)
            # output = self.softmax(output)
            outputs.append(logits.unsqueeze(1))  # shape [batch_size, 1, input_size]
        # concatenate across time
        outputs = torch.cat(outputs, dim=1)

        if return_states:
            if isinstance(self.cell, LSTMCell) or isinstance(self.cell, mLSTMCell):
                return outputs, h, cell_state
            else:
                return outputs, h
        else:
            return outputs

def plot_loss(losses, title, y_label, path):
    """
    Plots the loss over epochs and saves the plot to a file.

    Args:
        losses (list or array-like): A list or array containing the loss values for each epoch.
        title (str): The title for the plot.
        path (str): The file path where the plot image will be saved.
    """

    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(path)
    plt.close()

def generate_sequences(seq_length, padding, vocabulary, 
                       delimiter, unknown, output_len):
    """
    Generate a sequence of inputs and outputs for the copy task.

    The input sequence is a sequence of random elements from the vocabulary
    of length seq_length. Then padding number of delimiters are appended to the
    input sequence. Then unknown number of unknowns are appended to the input
    sequence. The output sequence is the same as the input sequence except the
    first output_padding elements are unknowns.

    Args:
        seq_length (int): The length of the sequence to generate.
        padding (int): The number of padding elements to add to the sequence.
        vocabulary (list): The list of elements to choose from for the sequence.
        delimiter (int): The id of the delimiter element.
        unknown (int): The id of the unknown element.
        output_len (int): The length of the output sequence.

    Returns:
        tuple: A tuple of two lists. The first list is the input sequence and
        the second list is the output sequence.
    """
    input_seq = []
    output = []
    for index in range(seq_length):
        input_seq.append(np.random.choice(vocabulary))
    output = input_seq.copy()
    for index in range(padding):
        input_seq.append(delimiter)
    output_padding = len(input_seq)
    for index in range(output_len):
        input_seq.append(unknown)
    output = output_padding * [unknown] + output[:output_len]
    return input_seq, output

    
if __name__ == '__main__':
    # accept command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=str, default='cpu',
                        help='mps, cuda, cpu')
    parser.add_argument('-l', '--log', type=str, default='rnn.log', 
                        help='filename.log')
    parser.add_argument('-iseq', '--train_seq_len', type=int, default=100, 
                        help='sequence length of train set')
    parser.add_argument('-tseq', '--test_seq_len', type=int, default=100, 
                        help='sequence length of test set')
    parser.add_argument('-p', '--padding', type=int, default=10, 
                        help='padding')
    parser.add_argument('-oseq', '--train_output_len', type=int, default=50, 
                        help='output length')
    parser.add_argument('-toseq', '--test_output_len', type=int, default=50, 
                        help='output length')
    parser.add_argument('-b', '--batch_size', type=int, default=32, 
                        help='batch size')
    parser.add_argument('-em', '--embedding_size', type=int, default=11, 
                        help='embedding size')
    parser.add_argument('-hs', '--hidden_size', type=int, default=128, 
                        help='hidden size')
    parser.add_argument('-c', '--cell', type=str, default='rnn', 
                        help='rnn/gru/lstm/mgru/mlstm')
    parser.add_argument('-ep', '--epochs', type=int, default=10, 
                        help='epochs')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.1, 
                        help='learning rate')
    parser.add_argument('-s', '--seed', type=int, default=42, 
                        help='seed')
    parser.add_argument('-trs', '--train_samples', type=int, default=10000, 
                        help='train samples')
    parser.add_argument('-ts', '--test_samples', type=int, default=1000, 
                        help='test samples')
    args = parser.parse_args()

    if not os.path.exists('logs'): os.makedirs('logs')
    log_dir = os.path.join("logs", args.log)
    if not os.path.exists(log_dir): os.makedirs(log_dir)

    # Configure the logger
    logger.add(os.path.join(log_dir, 'logs.log'), format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="INFO")

    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    # else:
    #     device = torch.device("cpu")
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    vocabulary = ['a','x','c','r','y','w','b','t','o']
    delimiter = '$'
    unknown = ' '
    vocabulary.extend([delimiter, unknown])
    print(f"vocabulary: {vocabulary}")
    
    # lookup table to convert characters to indices
    char2idx = {char: idx for idx, char in enumerate(vocabulary)}
    # lookup table to convert indices to characters
    idx2char = {idx: char for idx, char in enumerate(vocabulary)}
    # convert vocabulary to indices
    idx_vocab = [char2idx[char] for char in vocabulary]
    print(f"idx_vocab: {idx_vocab}")
    idx_vocab.remove(char2idx[delimiter])
    idx_vocab.remove(char2idx[unknown])

    padding = args.padding  # repeat delimiter for how many time steps
    batch_size = args.batch_size
    input_size = len(vocabulary)
    embedding_size = args.embedding_size
    hidden_size = args.hidden_size
    n_epochs = args.epochs
    lr = args.learning_rate
    train_samples = args.train_samples
    test_samples = args.test_samples

    np.random.seed(args.seed)

    train_size = args.train_seq_len
    test_size = args.test_seq_len
    train_output_len = args.train_output_len
    test_output_len = args.test_output_len

    # Generate train samples for copy task
    X_train = []
    Y_train = []
    tqdm.write(f"Generating {train_samples} train samples...")
    for index in tqdm(range(train_samples)):
        input_seq, output = generate_sequences(train_size, padding=padding, vocabulary=idx_vocab, delimiter=char2idx[delimiter], unknown=char2idx[unknown], output_len=train_output_len)
        X_train.append(input_seq)
        Y_train.append(output)
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    # pkl.dump((X_train, Y_train), open(f'copyTask_data_N{train_samples}_T{seq_length[0]}_P{padding[0]}_O{train_output_len}.pkl', 'wb'))

    # Generate test samples for copy task
    X_test = []
    Y_test = []
    tqdm.write(f"Generating {test_samples} test samples...")
    for index in tqdm(range(test_samples)):
        input_seq, output = generate_sequences(test_size, padding=padding, vocabulary=idx_vocab, delimiter=char2idx[delimiter], unknown=char2idx[unknown], output_len=test_output_len)
        X_test.append(input_seq)
        Y_test.append(output)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    # pkl.dump((X_test, Y_test), open(f'copyTask_data_N{train_samples}_T{seq_length[0]}_P{padding[0]}_O{test_output_len}.pkl', 'wb'))

    # (X_train, Y_train) = pkl.load(open('copyTask_data_N10000_T100_P10_O50.pkl', 'rb'))
    logger.info(f"Loaded train data with shape: {X_train.shape}, {Y_train.shape}")
    # (X_test, Y_test) = pkl.load(open('copyTask_data_N10000_T500_P50_O100.pkl', 'rb'))
    logger.info(f"Loaded test data with shape: {X_test.shape}, {Y_test.shape}")
    
    # Model
    model = RNN(len(vocabulary), embedding_size, 
                hidden_size, device, args.cell)
    
    # Load a saved model
    # model = torch.load('logs/<model_path>')

    # check whether parameters of RNN as well as the required cell are obtained
    for name, param in model.named_parameters():
        print(name, param.size())
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    logger.info(f"Vocabulary: {vocabulary}")
    logger.info(f"Unknown: {unknown}, Delimiter: {delimiter}")
    logger.info(f"sequence length: {train_size}, padding: {padding}, output length: {train_output_len}")
    logger.info(f"batch size: {batch_size}, input size: {input_size}, embedding size: {embedding_size}, hidden size: {hidden_size}")
    logger.info(f"epochs: {n_epochs}, lr: {lr}")
    logger.info(f"train samples: {train_samples}, test samples: {test_samples}")
    
    train_losses = []
    train_accuracies = []
    start_time = time.time()

    # Training
    for epoch in range(n_epochs):
        train_loss = 0.0
        train_accuracy = 0.0
        cell_state = torch.zeros(batch_size, hidden_size, device=device)
        # shuffle training data
        train_indices = torch.randperm(train_samples)
        total_samples = 0
        tqdm.write("Training batches...")
        for index in tqdm(range(0, train_samples, batch_size)):
            batch_indices = train_indices[index:index+batch_size]
            batchX = torch.tensor(X_train[batch_indices], 
                                 dtype=torch.long).to(device)
            batchY = torch.tensor(Y_train[batch_indices]).to(device)
            current_batch_size = len(batch_indices)
            if current_batch_size < batch_size:
                break
            total_samples += current_batch_size

            Y_one_hot = F.one_hot(batchY, input_size).float()
            optimizer.zero_grad()
            output = model(batchX) # [batch_size, seq_length, vocab_size]
            
            # Calculate loss at each time step
            time_step_losses = []
            for t in range(output.shape[1]):  # Iterate over sequence length
                time_step_output = output[:, t, :]  # [batch_size, vocab_size]
                time_step_target = Y_one_hot[:, t, :]  # [batch_size, vocab_size]
                time_step_loss = criterion(time_step_output, time_step_target)
                time_step_losses.append(time_step_loss)

            # Average time-step losses
            loss = torch.mean(torch.stack(time_step_losses))
            loss.backward()
            # clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            train_loss += loss.item() * current_batch_size
            
            # Calculate accuracy at each time step and average
            time_step_accuracies = []
            for t in range(output.shape[1]):
                time_step_output = F.softmax(output[:, t, :], dim=-1)
                time_step_target = Y_one_hot[:, t, :]
                accuracy = (torch.argmax(time_step_output, dim=-1) == torch.argmax(time_step_target, dim=-1)).float().mean()
                time_step_accuracies.append(accuracy)
            train_accuracy += torch.mean(torch.stack(time_step_accuracies)).item() * current_batch_size

            # output = output[:, -train_output_len:, :]
            # Y_one_hot = Y_one_hot[:, -train_output_len:, :]
            # # predictions = [Batch, Time, Output]
            # train_accuracy += (torch.argmax(output, dim=-1) == torch.argmax(Y_one_hot, dim=-1)).sum().item()
        train_loss /= total_samples
        train_losses.append(train_loss)
        train_accuracy /= total_samples
        train_accuracies.append(train_accuracy)
        # print(f"Epoch {epoch} | Training, Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")

        # Log train metrics
        logger.info(f"Epoch {epoch} | Loss torch: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
    
    logger.info(f"train time: {time.time()-start_time}")
    
    plot_loss(train_losses, "Training Loss", "Loss", os.path.join(log_dir, "train_loss.png"))
    plot_loss(train_accuracies, "Training Accuracy", "Accuracy", os.path.join(log_dir, "train_accuracy.png"))

    # Test the model
    model.eval()
    tqdm.write("Testing model...")
    test_loss = 0.0
    test_accuracy = 0.0
    start_time = time.time()
    total_samples = 0
    for index in range(0, test_samples, batch_size):
        batchX = torch.tensor(X_test[index:index+batch_size]).to(device=device)
        batchY = torch.tensor(Y_test[index:index+batch_size]).to(device=device)
        current_batch_size = len(batchX)
        if current_batch_size < batch_size:
            break
        total_samples += current_batch_size

        # If test data has more time steps than train data
        # Then we split the test data into multiple folds 
        # of the same size as the train data
        Y_one_hot = F.one_hot(batchY, input_size).float()
        predictions = torch.zeros_like(Y_one_hot)
        if test_size > train_size:
            # since input length is longer than train size, we need to preserve h and cell state (in case of lstm/mlstm)
            h = torch.zeros(batch_size, hidden_size, device=device)
            cell_state = torch.zeros(batch_size, hidden_size, device=device)
            for fold in range(0, test_size, train_size):
                X_fold = batchX[:, fold: fold+train_size]
                # Y_fold = batchY[:, fold: fold+train_size]
                if args.cell == "lstm" or args.cell == "mlstm":
                    output, h, cell_state = model(X_fold, h=h, cell_state=cell_state, return_states=True) # [batch_size, seq_length, input_size]
                else:
                    output, h = model(X_fold, h=h, return_states=True) # [batch_size, seq_length, input_size]
                # predictions[:, fold: fold+train_size, :] = output
                predictions[:, fold: fold+train_size, :] = output

            # Calculate loss at each time step
            time_step_losses = []
            for t in range(Y_one_hot.shape[1]):  # Iterate over sequence length
                time_step_output = predictions[:, t, :]  # [batch_size, vocab_size]
                time_step_target = Y_one_hot[:, t, :]  # [batch_size, vocab_size]
                time_step_loss = criterion(time_step_output, time_step_target)
                time_step_losses.append(time_step_loss)
            # Average time-step losses
            loss = torch.mean(torch.stack(time_step_losses))
            test_loss += loss.item() * current_batch_size
            
            # Calculate accuracy at each time step and average
            time_step_accuracies = []
            for t in range(predictions.shape[1]):
                time_step_output = F.softmax(predictions[:, t, :], dim=-1)
                time_step_target = Y_one_hot[:, t, :]
                accuracy = (torch.argmax(time_step_output, dim=-1) == torch.argmax(time_step_target, dim=-1)).float().mean()
                time_step_accuracies.append(accuracy)
            test_accuracy += torch.mean(torch.stack(time_step_accuracies)).item() * current_batch_size

            # loss = criterion(predictions, Y_one_hot)
            # predictions = predictions[:, -test_output_len:, :]
            # Y_one_hot = Y_one_hot[:, -test_output_len:, :]
            # accuracy = (torch.argmax(output, dim=-1) == torch.argmax(Y_one_hot, dim=-1)).sum().item()
            # test_loss += loss.item() * current_batch_size
            # test_accuracy += accuracy
    test_loss /= total_samples
    test_accuracy /= total_samples
    # print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    # print(f"test time: {time.time()-start_time}")
    logger.info(f"test time: {time.time()-start_time}")

    # Save model
    torch.save(model, os.path.join(log_dir, 'model.pth') )
