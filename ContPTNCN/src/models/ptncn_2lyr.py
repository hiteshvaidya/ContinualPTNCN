"""
Parallel Temporal Neural Coding Network (P-TNCN) - 2-Layer Implementation
========================================================================

Implementation of a 2-latent variable layer Parallel Temporal Neural Coding Network 
based on Ororbia and Mali 2019 IEEE TNNLS paper.

The P-TNCN implements a biologically-inspired neural architecture that combines:
1. Neural Coding Theory: Neurons encode information through temporal dynamics
2. Predictive Coding: Higher layers predict lower layer activities
3. Error-driven Learning: Learning is driven by prediction errors
4. Temporal Integration: Past states influence current processing

Key Components:
- Feedforward prediction weights (W1, W2): Generate predictions
- Error weights (E1, E2): Process prediction errors  
- Memory weights (M1, M2): Bottom-up information flow
- Recurrent weights (V1, V2): Temporal integration
- Control weights (U1): Top-down modulation

@author: Ankur Mali
"""

import tensorflow as tf
import sys
from utils import gelu1, create_dropout_mask, softmax, init_weights, gte, ltanh, d_relu6, standardize
import numpy as np

# Set random seeds for reproducible results
seed = 1234
tf.random.set_seed(seed=seed)
np.random.seed(seed)
class PTNCN:
    """
    Parallel Temporal Neural Coding Network (P-TNCN) - 2 Layer Architecture
    
    This class implements a biologically-inspired neural network based on neural coding
    theory and predictive coding principles. The network maintains temporal state and
    learns through error-driven dynamics.
    
    Architecture:
    - Layer 0: Input layer (x_dim dimensions)
    - Layer 1: First hidden layer (hid_dim dimensions) 
    - Layer 2: Second hidden layer (hid_dim dimensions)
    
    Key Features:
    - Temporal dynamics with recurrent connections
    - Predictive coding with top-down predictions
    - Error-driven learning rules
    - State-dependent processing
    """
    
    def __init__(self, name, x_dim, hid_dim, wght_sd=0.025, err_wght_sd=0.025, act_fun="tanh", 
                 init_type="normal", out_fun="identity", in_dim=-1, zeta=1.0):
        """
        Initialize P-TNCN model with specified architecture and parameters.
        
        Args:
            name: Model identifier string
            x_dim: Output/target dimension (vocabulary size for language modeling)
            hid_dim: Hidden layer dimensions for both layers 1 and 2
            wght_sd: Standard deviation for weight initialization
            err_wght_sd: Standard deviation for error weight initialization  
            act_fun: Activation function ("tanh", "relu", "sigmoid", etc.)
            init_type: Weight initialization method ("normal", "uniform", etc.)
            out_fun: Output layer activation ("softmax", "identity", etc.)
            in_dim: Input dimension (auto-determined if -1)
            zeta: Control parameter for top-down connections (1.0 = enabled)
        """
        # Basic model configuration
        self.name = name
        self.act_fun = act_fun
        self.hid_dim = hid_dim        # Hidden layer dimensions
        self.x_dim = x_dim            # Output dimension (e.g., vocabulary size)
        self.init_type = init_type
        self.zeta = zeta              # Top-down control strength (1.0 = full, 0.0 = none)
        
        # Model behavior flags
        self.L1 = 0                   # Lateral sparsity penalty (0 = disabled)
        self.standardize = False      # Input standardization (experimental)
        self.use_temporal_error_rule = False  # Temporal error dynamics (works better False for discrete inputs)
        
        # Determine input dimension
        if in_dim <= 0:
            in_dim = x_dim
        self.in_dim = in_dim

        ###########################################################################################################
        # Temporal State Variables - Neural Coding Network Dynamics
        ###########################################################################################################
        
        # These variables maintain the temporal state of the network across time steps
        # Critical for Neural Coding Network dynamics and sequence processing
        
        # Padding tensors for initialization
        self.zero_pad = tf.zeros([1, hid_dim])  # Template for zero padding
        
        # Current time step activations (zf = "z filtered" = post-activation)
        self.zf0 = None    # Layer 0 activation (input layer)
        self.zf1 = None    # Layer 1 activation (first hidden layer)  
        self.zf2 = None    # Layer 2 activation (second hidden layer)
        self.zf3 = None    # Layer 3 (unused in 2-layer version)
        self.zf4 = None    # Layer 4 (unused in 2-layer version)
        
        # Previous time step activations (tm1 = "t minus 1")
        # Essential for temporal integration and recurrent dynamics
        self.zf0_tm1 = None    # Previous input activation
        self.zf1_tm1 = None    # Previous layer 1 activation
        self.zf2_tm1 = None    # Previous layer 2 activation
        self.zf3_tm1 = None    # Previous layer 3 (unused)
        self.zf4_tm1 = None    # Previous layer 4 (unused)
        
        # Pre-activation states (z = raw layer inputs before activation)
        self.z1 = None     # Layer 1 pre-activation
        self.z2 = None     # Layer 2 pre-activation  
        self.z3 = None     # Layer 3 (unused)
        self.z4 = None     # Layer 4 (unused)
        
        # Target states (y = desired activations after error correction)
        self.y1 = None     # Layer 1 target activation
        self.y2 = None     # Layer 2 target activation
        self.y3 = None     # Layer 3 (unused)
        self.y4 = None     # Layer 4 (unused)
        
        # Error signals (e = prediction errors)
        self.ex = None     # Output/reconstruction error
        self.e1 = None     # Layer 1 prediction error  
        self.e2 = None     # Layer 2 prediction error
        self.e3 = None     # Layer 3 (unused)
        
        # Temporal error vectors (ev = error for learning dynamics)
        self.e1v = None        # Current layer 1 learning error
        self.e2v = None        # Current layer 2 learning error
        self.e3v = None        # Layer 3 (unused)
        self.e4v = None        # Layer 4 (unused)
        self.e1v_tm1 = None    # Previous layer 1 learning error
        self.e2v_tm1 = None    # Previous layer 2 learning error
        self.e3v_tm1 = None    # Previous layer 3 (unused)
        self.e4v_tm1 = None    # Previous layer 4 (unused)
        
        # Additional state variables
        self.x = None          # Current input
        self.z_pad = None      # Batch-specific zero padding for hidden layers
        self.x_pad = None      # Batch-specific zero padding for input layer
        self.x_tm1 = None      # Previous input (unused currently)

        ###########################################################################################################
        # Weight Matrix Definitions - P-TNCN Architecture
        ###########################################################################################################

        ###########################################################################################################
        # Weight Matrix Definitions - P-TNCN Architecture
        ###########################################################################################################
        
        # Top-down control weights (U matrices) - Optional hierarchical control
        if self.zeta > 0.0:
            # U1: Layer 2 → Layer 1 top-down control connections
            # Allows higher layers to modulate lower layer processing
            self.U1 = tf.Variable(init_weights(self.init_type, [self.hid_dim, self.hid_dim], 
                                             stddev=wght_sd, seed=seed))
        
        # Bottom-up data-driving weights (M matrices) - Information flow upward
        # M2: Layer 1 → Layer 2 bottom-up connections
        self.M2 = tf.Variable(init_weights(self.init_type, [self.hid_dim, self.hid_dim], 
                                         stddev=wght_sd, seed=seed))
        # M1: Input → Layer 1 bottom-up connections  
        self.M1 = tf.Variable(init_weights(self.init_type, [in_dim, self.hid_dim], 
                                         stddev=wght_sd, seed=seed))

        # Top-down prediction weights (W matrices) - Predictive coding
        # W2: Layer 2 → Layer 1 prediction (what layer 2 expects layer 1 to be)
        self.W2 = tf.Variable(init_weights(self.init_type, [self.hid_dim, self.hid_dim], 
                                         stddev=wght_sd, seed=seed))
        # W1: Layer 1 → Output prediction (what layer 1 predicts for output)
        self.W1 = tf.Variable(init_weights(self.init_type, [self.hid_dim, self.x_dim], 
                                         stddev=wght_sd, seed=seed))

        # Recurrent temporal memory weights (V matrices) - Temporal integration
        # V2: Layer 2 self-recurrent connections (temporal memory)
        self.V2 = tf.Variable(init_weights(self.init_type, [self.hid_dim, self.hid_dim], 
                                         stddev=wght_sd, seed=seed))
        # V1: Layer 1 self-recurrent connections (temporal memory)
        self.V1 = tf.Variable(init_weights(self.init_type, [self.hid_dim, self.hid_dim], 
                                         stddev=wght_sd, seed=seed))

        # Bottom-up error weights (E matrices) - Error-driven learning
        # E2: Layer 1 error → Layer 2 error propagation
        self.E2 = tf.Variable(init_weights(self.init_type, [self.hid_dim, self.hid_dim], 
                                         stddev=err_wght_sd, seed=seed))
        # E1: Output error → Layer 1 error propagation
        self.E1 = tf.Variable(init_weights(self.init_type, [self.x_dim, self.hid_dim], 
                                         stddev=err_wght_sd, seed=seed))

        ###########################################################################################################
        # Parameter Organization for TensorFlow Optimizers
        ###########################################################################################################
        
        # Organize all learnable parameters for TensorFlow optimizer compatibility
        # Order matters for gradient application in compute_updates()
        self.param_var = []  # List of all trainable parameters
        
        # Prediction weights
        self.param_var.append(self.W1)  # Layer 1 → Output predictions
        self.param_var.append(self.E1)  # Output → Layer 1 error weights
        self.param_var.append(self.W2)  # Layer 2 → Layer 1 predictions  
        self.param_var.append(self.E2)  # Layer 1 → Layer 2 error weights
        
        # Top-down control (if enabled)
        if self.zeta > 0.0:
            self.param_var.append(self.U1)  # Layer 2 → Layer 1 control
            
        # Bottom-up information flow
        self.param_var.append(self.M1)  # Input → Layer 1
        self.param_var.append(self.M2)  # Layer 1 → Layer 2
        
        # Temporal recurrent connections  
        self.param_var.append(self.V1)  # Layer 1 temporal memory
        self.param_var.append(self.V2)  # Layer 2 temporal memory

        ###########################################################################################################
        # Activation Function Setup
        ###########################################################################################################
        ###########################################################################################################
        # Activation Function Setup
        ###########################################################################################################
        
        # Configure hidden layer activation function
        self.act_fx = None
        if self.act_fun == "gelu":
            self.act_fx = gelu1                # Gaussian Error Linear Unit
        elif self.act_fun == "relu6":
            self.act_fx = tf.nn.relu6          # ReLU capped at 6
        elif self.act_fun == "relu":
            self.act_fx = tf.nn.relu           # Rectified Linear Unit
        elif self.act_fun == "sigmoid":
            self.act_fx = tf.nn.sigmoid        # Sigmoid (0,1) output
        elif self.act_fun == "sign":
            self.act_fx = tf.sign              # Sign function (-1,1)
        elif self.act_fun == "tanh":
            self.act_fx = tf.nn.tanh           # Hyperbolic tangent (-1,1)
        elif self.act_fun == "ltanh":
            self.act_fx = ltanh                # Leaky tanh variant
        else:
            self.act_fx = gte                  # Greater-than-equal threshold

        # Configure output layer activation function
        self.out_fx = tf.identity              # Default: linear output
        if out_fun == "softmax":
            self.out_fx = softmax              # Probability distribution (for classification)
        elif out_fun == "tanh":
            self.out_fx = tf.nn.tanh           # Bounded output (-1,1)
        elif out_fun == "sigmoid":
            self.out_fx = tf.nn.sigmoid        # Bounded output (0,1)

    def act_dx(self, h, z):
        """
        Compute derivative of activation function for gradient computation.
        
        Args:
            h: Pre-activation values (raw inputs to activation function)
            z: Post-activation values (outputs of activation function)
            
        Returns:
            Derivative of activation function at given points
            
        Note: Used for analytical gradient computation in learning rules.
              Currently supports tanh, ltanh, relu6, and relu derivatives.
        """
        if self.act_fun == "tanh":  
            # d/dh tanh(h) = 1 - tanh²(h) = 1 - z²
            return -(z * z) + 1.0
        elif self.act_fun == "ltanh":  
            # Leaky tanh derivative
            return d_ltanh(z)
        elif self.act_fun == "relu6":
            # ReLU6 derivative (1 if 0 < h < 6, else 0)
            return d_relu6(h)
        elif self.act_fun == "relu":
            # ReLU derivative (1 if h > 0, else 0) 
            return d_relu(h)
        else:
            print("ERROR: deriv/dx fun not specified:{0}".format(self.act_fun))
            sys.exit(0)

    def collect_params(self):
        """
        Collect all model parameters in a named dictionary.
        
        Returns:
            theta: Dictionary mapping parameter names to weight matrices
            
        Purpose: Used for parameter norm monitoring, saving/loading models,
                and debugging weight magnitudes during training.
        """
        theta = dict()
        # Prediction weights
        theta["W1"] = self.W1  # Layer 1 → Output predictions
        theta["W2"] = self.W2  # Layer 2 → Layer 1 predictions
        
        # Error propagation weights
        theta["E1"] = self.E1  # Output → Layer 1 error weights
        theta["E2"] = self.E2  # Layer 1 → Layer 2 error weights
        
        # Bottom-up information flow
        theta["M1"] = self.M1  # Input → Layer 1
        theta["M2"] = self.M2  # Layer 1 → Layer 2
        
        # Top-down control (if enabled)
        if self.zeta > 0.0:
            theta["U1"] = self.U1  # Layer 2 → Layer 1 control
            
        # Temporal recurrent connections
        theta["V1"] = self.V1  # Layer 1 temporal memory
        theta["V2"] = self.V2  # Layer 2 temporal memory
        
        return theta

    def get_complexity(self):
        """
        Calculate model complexity in terms of total number of parameters.
        
        Returns:
            wght_cnt: Total number of learnable parameters (synaptic weights)
            
        Purpose: Used for model comparison and computational complexity analysis.
        """
        wght_cnt = 0
        for i in range(len(self.param_var)):
            # Get dimensions of each parameter matrix
            wr = self.param_var[i].shape[0]  # Number of rows
            wc = self.param_var[i].shape[1]  # Number of columns
            wght_cnt += (wr * wc)            # Add total elements
        return wght_cnt

    def forward(self, x, m, K=5, beta=0.2, alpha=1, is_eval=True):
        """
        Forward pass through P-TNCN network with Neural Coding Network dynamics.
        
        Args:
            x: Input tensor [batch_size, input_dim] (one-hot encoded tokens)
            m: Mask tensor [batch_size, 1] (1 for valid tokens, 0 for padding)
            K: Number of iterations for settling dynamics (unused, kept for compatibility)
            beta: Temporal integration strength (controls how much past influences present)
            alpha: Error-driven learning strength (controls error integration)
            is_eval: Whether in evaluation mode (affects learning dynamics)
            
        Returns:
            x_logits: Raw output predictions [batch_size, output_dim]
            x_mu: Probability distribution over output [batch_size, output_dim] 
            
        Process:
        1. Initialize/update temporal state variables
        2. Compute layer activations with temporal integration
        3. Generate predictions using top-down weights
        4. Calculate prediction errors at each layer
        5. Compute target states using error correction
        6. Update temporal error signals for learning
        """
        # Initialize return variables
        y_logits = None
        y_mu = None
        x_logits = None
        x_mu = None
        x_ = tf.cast(x, dtype=tf.float32)  # Ensure float32 for computations

        ###########################################################################################################
        # Temporal State Management
        ###########################################################################################################
        
        # Update previous time step states (critical for temporal dynamics)
        if self.zf1 is not None:
            # If we have previous states, store them as t-1 (previous time step)
            self.zf0_tm1 = self.zf0              # Previous input activation
            if self.x_tm1 is not None:
                self.zf0_tm1 = self.x_tm1        # Use stored previous input if available
            self.zf1_tm1 = self.y1               # Previous layer 1 activation (target state)
            self.zf2_tm1 = self.y2               # Previous layer 2 activation (target state)
            self.e1v_tm1 = self.e1v              # Previous layer 1 learning error
            self.e2v_tm1 = self.e2v              # Previous layer 2 learning error
        else:
            # First time step: initialize with zeros
            # Create batch-appropriate zero padding
            if self.z_pad is None:
                self.z_pad = tf.zeros([x.shape[0], self.hid_dim])   # Hidden layer zeros
                self.x_pad = tf.zeros([x.shape[0], self.in_dim])    # Input layer zeros
            else:
                # Adjust padding if batch size changed
                if self.z_pad.shape[0] != x.shape[0]:
                    self.z_pad = tf.zeros([x.shape[0], self.hid_dim])
                    self.x_pad = tf.zeros([x.shape[0], self.in_dim])
            
            # Initialize all temporal states to zero
            self.zf0_tm1 = self.x_pad    # No previous input
            self.zf1_tm1 = self.z_pad    # No previous layer 1 activation
            self.zf2_tm1 = self.z_pad    # No previous layer 2 activation
            self.e1v_tm1 = self.z_pad    # No previous layer 1 error
            self.e2v_tm1 = self.z_pad    # No previous layer 2 error
            
            # Initialize current states to zero
            self.z1 = self.z_pad         # Layer 1 pre-activation
            self.z2 = self.z_pad         # Layer 2 pre-activation
            self.z3 = self.z_pad         # Layer 3 (unused)
            self.e1 = self.z_pad         # Layer 1 error
            self.e2 = self.z_pad         # Layer 2 error
            self.ex = self.zf0_tm1       # Output error

        # Set current input state
        self.zf0 = x_  # Current input activation

        ###########################################################################################################
        # Layer 2 Computation (Second Hidden Layer)
        ###########################################################################################################
        
        # Compute layer 2 pre-activation: combine bottom-up and temporal information
        if self.zeta > 0.0:
            # With top-down control: M2(layer1→layer2) + V2(temporal_memory)
            self.z2 = tf.add(tf.matmul(self.zf1_tm1, self.M2), tf.matmul(self.zf2_tm1, self.V2))
        else:
            # Without top-down control: only temporal memory
            self.z2 = tf.matmul(self.zf2_tm1, self.V2)
        
        # Optional input standardization (experimental)
        if self.standardize is True:
            self.z2 = standardize(self.z2)
            
        # Apply activation function to get layer 2 activation
        self.zf2 = self.act_fx(self.z2)
        
        # Generate layer 2's prediction for layer 1 (top-down prediction)
        z1_mu = tf.matmul(self.zf2, self.W2)

        ###########################################################################################################
        # Layer 1 Computation (First Hidden Layer)  
        ###########################################################################################################
        
        # Compute layer 1 pre-activation: combine input, top-down control, and temporal memory
        if self.zeta > 0.0:
            # Full connectivity: M1(input→layer1) + U1(layer2→layer1) + V1(temporal_memory)
            self.z1 = tf.add(tf.add(tf.matmul(self.zf0_tm1, self.M1), 
                                   tf.matmul(self.zf2_tm1, self.U1)), 
                            tf.matmul(self.zf1_tm1, self.V1))
        else:
            # Without top-down control: M1(input→layer1) + V1(temporal_memory)
            self.z1 = tf.add(tf.matmul(self.zf0_tm1, self.M1), tf.matmul(self.zf1_tm1, self.V1))
        
        # Optional input standardization
        if self.standardize is True:
            self.z1 = standardize(self.z1)
            
        # Apply activation function to get layer 1 activation
        self.zf1 = self.act_fx(self.z1)
        
        # Generate final output predictions
        x_logits = tf.matmul(self.zf1, self.W1)        # Raw predictions
        x_mu = self.out_fx(x_logits)                   # Apply output activation (e.g., softmax)

        ###########################################################################################################
        # Error Computation and Neural Coding Dynamics
        ###########################################################################################################
        
        # Compute prediction errors at each layer (key to Neural Coding Network)
        # Layer 1 error: difference between layer 2's prediction and actual layer 1 activation
        self.e1 = tf.subtract(z1_mu, self.zf1) * m      # Masked by valid tokens
        # Output error: difference between layer 1's prediction and actual input
        self.ex = tf.subtract(x_mu, self.zf0) * m       # Masked by valid tokens
        
        # Compute error-driven perturbations for each layer
        # d2: Error signal for layer 2 (propagated from layer 1 error)
        d2 = tf.matmul(self.e1, self.E2)
        # d1: Error signal for layer 1 (from output error minus scaled layer 1 error)
        d1 = tf.subtract(tf.matmul(self.ex, self.E1), self.e1 * alpha)
        
        # Optional sparsity regularization (L1 penalty on activations)
        if self.L1 > 0.0:
            # Add sparsity pressure: penalize large activations
            d2 = tf.add(d2, tf.sign(self.z2) * self.L1)  # Layer 2 sparsity
            d1 = tf.add(d1, tf.sign(self.z1) * self.L1)  # Layer 1 sparsity
        
        # Compute target states using error correction (Neural Coding Network dynamics)
        # Target = current_activation - beta * error_signal
        # beta controls how much the error influences the target (temporal integration)
        self.y2 = self.act_fx(tf.subtract(self.z2, d2 * beta))  # Layer 2 target state
        self.y1 = self.act_fx(tf.subtract(self.z1, d1 * beta))  # Layer 1 target state
        
        # Compute temporal error signals for learning (difference between actual and target)
        # These drive the learning dynamics in compute_updates()
        self.e2v = tf.subtract(self.zf2, self.y2) * m   # Layer 2 learning error
        self.e1v = tf.subtract(self.zf1, self.y1) * m   # Layer 1 learning error

        # Note: Masking is applied to errors, not activations, for efficiency
        # The mask ensures that padding tokens don't contribute to learning

        return x_logits, x_mu

    def compute_updates(self, m, gamma=1.0, update_radius=-1.0):
        """
        Compute weight updates using P-TNCN's error-driven learning rules.
        
        Args:
            m: Mask tensor [batch_size, 1] for valid tokens
            gamma: Learning rate scaling factor for error weights
            update_radius: Gradient clipping radius (-1 = no clipping)
            
        Returns:
            delta_list: List of weight updates in same order as self.param_var
            
        P-TNCN Learning Rules:
        1. Prediction weights (W): Learn to minimize prediction errors
        2. Error weights (E): Learn to propagate errors effectively  
        3. Memory weights (M): Learn bottom-up information flow
        4. Recurrent weights (V): Learn temporal dependencies
        5. Control weights (U): Learn top-down modulation
        
        The learning is driven by temporal error signals (e1v, e2v) computed
        in the forward pass, implementing Neural Coding Network dynamics.
        """
        delta_list = []
        ###########################################################################################################
        # Weight Update Computation - P-TNCN Learning Rules
        ###########################################################################################################
        
        # W1: Layer 1 → Output prediction weights
        # Learn to minimize output reconstruction error
        dW = tf.matmul(self.zf1, self.ex, transpose_a=True)  # zf1^T * ex
        if update_radius > 0.0:
            dW = tf.clip_by_norm(dW, update_radius)
        delta_list.append(dW)

        # E1: Output → Layer 1 error propagation weights
        # Learn to effectively propagate output errors to layer 1
        if self.use_temporal_error_rule is True:
            # Temporal error rule: learn from changes in error over time
            dW = tf.matmul(self.ex, tf.subtract(self.e1v, self.e1v_tm1), transpose_a=True) * -gamma
        else:
            # Standard rule: transpose of prediction weight update scaled by gamma
            dW = tf.transpose(dW) * gamma
        if update_radius > 0.0:
            dW = tf.clip_by_norm(dW, update_radius)
        delta_list.append(dW)

        # W2: Layer 2 → Layer 1 prediction weights  
        # Learn to minimize layer 1 prediction error
        dW = tf.matmul(self.zf2, self.e1v, transpose_a=True)  # zf2^T * e1v
        if update_radius > 0.0:
            dW = tf.clip_by_norm(dW, update_radius)
        delta_list.append(dW)

        # E2: Layer 1 → Layer 2 error propagation weights
        # Learn to effectively propagate layer 1 errors to layer 2
        if self.use_temporal_error_rule is True:
            # Temporal error rule: learn from changes in error over time
            dW = tf.matmul(self.e1, tf.subtract(self.e2v, self.e2v_tm1), transpose_a=True) * -gamma
        else:
            # Standard rule: transpose of prediction weight update scaled by gamma
            dW = tf.transpose(dW) * gamma
        if update_radius > 0.0:
            dW = tf.clip_by_norm(dW, update_radius)
        delta_list.append(dW)

        # U1: Layer 2 → Layer 1 top-down control weights (if enabled)
        if self.zeta > 0.0:
            # Learn top-down modulation to minimize layer 1 errors
            dW = tf.matmul(self.zf2_tm1, self.e1v, transpose_a=True)  # zf2_tm1^T * e1v
            if update_radius > 0.0:
                dW = tf.clip_by_norm(dW, update_radius)
            delta_list.append(dW)

        # M1: Input → Layer 1 bottom-up weights
        # Learn to process input information to minimize layer 1 errors
        dW = tf.matmul(self.zf0_tm1, self.e1v, transpose_a=True)  # zf0_tm1^T * e1v
        if update_radius > 0.0:
            dW = tf.clip_by_norm(dW, update_radius)
        delta_list.append(dW)

        # M2: Layer 1 → Layer 2 bottom-up weights
        # Learn to process layer 1 information to minimize layer 2 errors
        dW = tf.matmul(self.zf1_tm1, self.e2v, transpose_a=True)  # zf1_tm1^T * e2v
        if update_radius > 0.0:
            dW = tf.clip_by_norm(dW, update_radius)
        delta_list.append(dW)

        # V1: Layer 1 temporal recurrent weights
        # Learn temporal dependencies to minimize layer 1 errors
        dW = tf.matmul(self.zf1_tm1, self.e1v, transpose_a=True)  # zf1_tm1^T * e1v
        if update_radius > 0.0:
            dW = tf.clip_by_norm(dW, update_radius)
        delta_list.append(dW)

        # V2: Layer 2 temporal recurrent weights
        # Learn temporal dependencies to minimize layer 2 errors
        dW = tf.matmul(self.zf2_tm1, self.e2v, transpose_a=True)  # zf2_tm1^T * e2v
        if update_radius > 0.0:
            dW = tf.clip_by_norm(dW, update_radius)
        delta_list.append(dW)

        return delta_list

    def clear_var_history(self):
        """
        Reset all temporal state variables to None for processing new sequences.
        
        Purpose: P-TNCN maintains temporal state across time steps within a sequence,
                but this state should be cleared between independent sequences to
                prevent information leakage between unrelated sequences.
                
        Called: After processing each complete sequence in training/evaluation.
        
        Critical for: Proper sequence independence in language modeling tasks
                     where each sequence should be processed independently.
        """
        # Current time step activations
        self.zf0 = None    # Input layer activation
        self.zf1 = None    # Layer 1 activation
        self.zf2 = None    # Layer 2 activation
        self.zf3 = None    # Layer 3 (unused)
        self.zf4 = None    # Layer 4 (unused)
        
        # Previous time step activations  
        self.zf0_tm1 = None    # Previous input activation
        self.zf1_tm1 = None    # Previous layer 1 activation
        self.zf2_tm1 = None    # Previous layer 2 activation
        self.zf3_tm1 = None    # Previous layer 3 (unused)
        self.zf4_tm1 = None    # Previous layer 4 (unused)
        
        # Pre-activation states
        self.z1 = None     # Layer 1 pre-activation
        self.z2 = None     # Layer 2 pre-activation
        self.z3 = None     # Layer 3 (unused)
        self.z4 = None     # Layer 4 (unused)
        
        # Target states (from error correction)
        self.y1 = None     # Layer 1 target activation
        self.y2 = None     # Layer 2 target activation
        self.y3 = None     # Layer 3 (unused)
        self.y4 = None     # Layer 4 (unused)
        
        # Error signals
        self.ex = None     # Output reconstruction error
        self.e1 = None     # Layer 1 prediction error
        self.e2 = None     # Layer 2 prediction error
        self.e3 = None     # Layer 3 (unused)
        
        # Learning error signals
        self.e1v = None        # Current layer 1 learning error
        self.e2v = None        # Current layer 2 learning error
        self.e3v = None        # Layer 3 (unused)
        self.e4v = None        # Layer 4 (unused)
        self.e1v_tm1 = None    # Previous layer 1 learning error
        self.e2v_tm1 = None    # Previous layer 2 learning error
        self.e3v_tm1 = None    # Previous layer 3 (unused)
        self.e4v_tm1 = None    # Previous layer 4 (unused)
        
        # Additional state
        self.x = None          # Current input storage
