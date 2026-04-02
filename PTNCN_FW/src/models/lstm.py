import tensorflow as tf
import numpy as np
from utils import init_weights

seed = 1234
tf.random.set_seed(seed=seed)
np.random.seed(seed)

class LSTM:
    def __init__(self, name, x_dim, hid_dim, wght_sd=0.025, init_type="normal"):
        self.name = name
        self.hid_dim = hid_dim
        self.x_dim = x_dim
        self.init_type = init_type
        
        # LSTM state variables
        self.h = None  # hidden state
        self.c = None  # cell state
        self.h_tm1 = None  # previous hidden state
        self.c_tm1 = None  # previous cell state
        
        # Initialize LSTM weights
        # Input gate weights
        self.W_xi = tf.Variable(init_weights(init_type, [x_dim, hid_dim], stddev=wght_sd, seed=seed))
        self.W_hi = tf.Variable(init_weights(init_type, [hid_dim, hid_dim], stddev=wght_sd, seed=seed))
        self.b_i = tf.Variable(tf.zeros([hid_dim]))
        
        # Forget gate weights
        self.W_xf = tf.Variable(init_weights(init_type, [x_dim, hid_dim], stddev=wght_sd, seed=seed))
        self.W_hf = tf.Variable(init_weights(init_type, [hid_dim, hid_dim], stddev=wght_sd, seed=seed))
        self.b_f = tf.Variable(tf.zeros([hid_dim]))
        
        # Cell state weights
        self.W_xc = tf.Variable(init_weights(init_type, [x_dim, hid_dim], stddev=wght_sd, seed=seed))
        self.W_hc = tf.Variable(init_weights(init_type, [hid_dim, hid_dim], stddev=wght_sd, seed=seed))
        self.b_c = tf.Variable(tf.zeros([hid_dim]))
        
        # Output gate weights
        self.W_xo = tf.Variable(init_weights(init_type, [x_dim, hid_dim], stddev=wght_sd, seed=seed))
        self.W_ho = tf.Variable(init_weights(init_type, [hid_dim, hid_dim], stddev=wght_sd, seed=seed))
        self.b_o = tf.Variable(tf.zeros([hid_dim]))
        
        # Output projection weights
        self.W_hy = tf.Variable(init_weights(init_type, [hid_dim, x_dim], stddev=wght_sd, seed=seed))
        self.b_y = tf.Variable(tf.zeros([x_dim]))
        
        # Collect parameters for optimization
        self.param_var = [
            self.W_xi, self.W_hi, self.b_i,
            self.W_xf, self.W_hf, self.b_f,
            self.W_xc, self.W_hc, self.b_c,
            self.W_xo, self.W_ho, self.b_o,
            self.W_hy, self.b_y
        ]
    
    def forward(self, x, m=None):
        """
        Forward pass of the LSTM
        Args:
            x: input tensor of shape [batch_size, x_dim]
            m: optional mask tensor of shape [batch_size, 1]
        Returns:
            y: output tensor of shape [batch_size, x_dim]
        """
        x_ = tf.cast(x, dtype=tf.float32)
        
        # Initialize states if first time step
        if self.h is None:
            batch_size = x_.shape[0]
            self.h = tf.zeros([batch_size, self.hid_dim])
            self.c = tf.zeros([batch_size, self.hid_dim])
            self.h_tm1 = tf.zeros([batch_size, self.hid_dim])
            self.c_tm1 = tf.zeros([batch_size, self.hid_dim])
        
        # Update previous states
        self.h_tm1 = self.h
        self.c_tm1 = self.c
        
        # Apply mask to input if provided
        if m is not None:
            x_ = x_ * m
        
        # Input gate
        i = tf.sigmoid(
            tf.matmul(x_, self.W_xi) + 
            tf.matmul(self.h_tm1, self.W_hi) + 
            self.b_i
        )
        
        # Forget gate
        f = tf.sigmoid(
            tf.matmul(x_, self.W_xf) + 
            tf.matmul(self.h_tm1, self.W_hf) + 
            self.b_f
        )
        
        # Cell state
        c_tilde = tf.tanh(
            tf.matmul(x_, self.W_xc) + 
            tf.matmul(self.h_tm1, self.W_hc) + 
            self.b_c
        )
        self.c = f * self.c_tm1 + i * c_tilde
        
        # Output gate
        o = tf.sigmoid(
            tf.matmul(x_, self.W_xo) + 
            tf.matmul(self.h_tm1, self.W_ho) + 
            self.b_o
        )
        
        # Hidden state
        self.h = o * tf.tanh(self.c)
        
        # Output projection
        y = tf.matmul(self.h, self.W_hy) + self.b_y
        
        # Apply mask to output if provided
        if m is not None:
            y = y * m
        
        return y
    
    def collect_params(self):
        """
        Collect all parameters in a dictionary
        """
        theta = {
            "W_xi": self.W_xi, "W_hi": self.W_hi, "b_i": self.b_i,
            "W_xf": self.W_xf, "W_hf": self.W_hf, "b_f": self.b_f,
            "W_xc": self.W_xc, "W_hc": self.W_hc, "b_c": self.b_c,
            "W_xo": self.W_xo, "W_ho": self.W_ho, "b_o": self.b_o,
            "W_hy": self.W_hy, "b_y": self.b_y
        }
        return theta
    
    def get_complexity(self):
        """
        Calculate the total number of parameters
        """
        wght_cnt = 0
        for param in self.param_var:
            if len(param.shape) == 2:
                wght_cnt += param.shape[0] * param.shape[1]
            else:
                wght_cnt += param.shape[0]
        return wght_cnt
    
    def clear_var_history(self):
        """
        Clear the state variables
        """
        self.h = None
        self.c = None
        self.h_tm1 = None
        self.c_tm1 = None
