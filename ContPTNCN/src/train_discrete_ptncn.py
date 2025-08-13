"""
Predictive Temporal Neural Coding Network (P-TNCN) Training Script
================================================================

Trains a P-TNCN as a discrete token prediction model for character-level language modeling.
This implementation is based on the model proposed in Ororbia et al., 2019 IEEE TNNLS.

The P-TNCN combines principles from neural coding theory with predictive learning,
creating a biologically-inspired architecture for sequence modeling tasks.

Key Features:
- Character-level language modeling on Penn Treebank dataset
- Neural Coding Network (NCN) dynamics with temporal integration
- Error-driven learning with specialized update rules
- Comprehensive evaluation and logging system

@author: Ankur Mali
"""

import os
import time
import sys
import pickle
import math

# GPU Configuration - restrict to device 0 for consistent memory usage
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Optional: set GPU ordering by PCI bus
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # Use only GPU device 0

# Add local module directories to Python path for imports
sys.path.insert(0, 'models/')  # Contains P-TNCN model implementation
sys.path.insert(0, 'utils/')   # Contains data loading and utility functions

# Core dependencies
import tensorflow as tf
import numpy as np

# Custom modules for P-TNCN implementation
from ptncn_2lyr import PTNCN        # Main P-TNCN model class
from data import Vocab              # Vocabulary management
from seq_sampler import DataLoader  # Sequential data loading for training

# Set random seeds for reproducible results across runs
seed = 1234
tf.random.set_seed(seed=seed)  # TensorFlow random seed
np.random.seed(seed)           # NumPy random seed

###########################################################################################################
# Helper Functions for Training and Evaluation
###########################################################################################################

def seq_to_tokens(idx_seq, vocab, mb_idx=0):
    """
    Convert integer sequence indices back to human-readable string tokens.
    
    Args:
        idx_seq: Tensor of shape [seq_len, batch_size] containing token indices
        vocab: Vocabulary object with idx2token method
        mb_idx: Mini-batch index to extract (default: 0, first sequence in batch)
    
    Returns:
        tok_seq: String representation of the sequence with spaces between tokens
    
    Usage: Primarily for debugging and visualizing model predictions
    """
    tok_seq = ""
    for i in range(0, idx_seq.shape[0]):
        idx = idx_seq[i][mb_idx].numpy()  # Extract index at position i
        tok_seq += vocab.idx2token(idx) + " "  # Convert to character and add space
    return tok_seq

def theta_norms_to_str(model, is_header=False):
    """
    Generate string representation of model parameter norms for monitoring training dynamics.
    
    Args:
        model: P-TNCN model instance with collect_params() method
        is_header: If True, return parameter names; if False, return norms
    
    Returns:
        str: Comma-separated values of parameter names or their Euclidean norms
    
    Purpose: Track parameter magnitudes to detect gradient explosion/vanishing
    """
    str = ""
    theta = model.collect_params()  # Get all model parameters
    for param_name in theta:
        if is_header is False:
            # Calculate Euclidean norm of parameter tensor
            str += "{0}".format(tf.norm(theta[param_name], ord="euclidean"))
        else:
            # Just add parameter name for CSV header
            str += "{0}".format(param_name)
        str += ","
    str = str[:-1]  # Remove trailing comma
    return str

def create_fixed_point(data_set, n_rounds=1, n_seq_total=-1):
    """
    Create a fixed-point sequence sample for stable loss tracking during evaluation.
    
    Args:
        data_set: DataLoader object providing sequences
        n_rounds: Number of complete passes through dataset (if n_seq_total <= 0)
        n_seq_total: Specific number of sequences to collect (overrides n_rounds if > 0)
    
    Returns:
        samp_seq_list: List of tuples (tr, ip, sg, mk) representing fixed sequences
    
    Purpose: Creates a consistent evaluation set to track training progress without
             the variability introduced by random sampling
    """
    samp_seq_list = []
    n_seq = 0
    
    if n_seq_total > 0:
        # Collect specific number of sequences
        flag = True
        while flag:
            for tr, ip, sg, mk, nxt_snt_flag in data_set:
                samp_seq_list.append((tr, ip, sg, mk))
                n_seq += 1
                if n_seq >= n_seq_total:
                    flag = False
                    break
    else:
        # Collect sequences for specified number of rounds
        debug_n_rounds = -1
        for r in range(n_rounds):
            n_seq = 0
            for tr, ip, sg, mk, nxt_snt_flag in data_set:
                if debug_n_rounds <= 0:
                    samp_seq_list.append((tr, ip, sg, mk))
                else:
                    if r < debug_n_rounds:
                        samp_seq_list.append((tr, ip, sg, mk))
    return samp_seq_list

def fast_log_loss(probs, y_ind):
    """
    Efficient categorical negative log-likelihood computation using sparse indexing.
    
    Args:
        probs: Tensor of shape [batch_size, vocab_size] with prediction probabilities
        y_ind: Tensor of shape [batch_size, 1] with target token indices
    
    Returns:
        loss: Negative log-likelihood (scalar)
    
    Optimization: Instead of computing full one-hot encoding and matrix multiplication,
                 directly index the probability of the correct token for efficiency.
    
    Note: This is the "Ororbia-Mali method" for fast sparse categorical cross-entropy.
    """
    loss = 0.0
    py = probs.numpy()  # Convert to numpy for indexing
    
    for i in range(0, y_ind.shape[0]):
        ti = y_ind[i][0]  # Get target index for sequence position i
        if ti >= 0:  # Valid token (non-negative indices, excludes padding)
            py = probs[i, ti]  # Probability of correct token
            if py <= 0.0:
                py = 1e-8  # Numerical stability: avoid log(0)
            loss += np.log(py)  # Accumulate log probability
    
    return -loss  # Return negative log-likelihood

def eval_model(model, data_set, debug_step_print=False):
    """
    Comprehensive model evaluation on a fixed dataset.
    
    Args:
        model: P-TNCN model instance
        data_set: DataLoader or list of sequences for evaluation
        debug_step_print: If True, print detailed step information (unused)
    
    Returns:
        cost: Average negative log-likelihood per token
        acc: Average token-level accuracy
        ppl: Perplexity (bits-per-character if calc_bpc=True, else exp(cost))
        mse: Mean squared error (currently unused, returns 0)
    
    Process:
        1. Iterate through all sequences in evaluation set
        2. For each sequence, step through time positions
        3. Convert tokens to one-hot, run forward pass
        4. Accumulate loss and accuracy starting from t_prime
        5. Normalize metrics by total number of tokens
    """
    # Initialize evaluation metrics
    cost = 0.0      # Accumulated negative log-likelihood
    acc = 0.0       # Accumulated correct predictions
    mse = 0.0       # Mean squared error (unused)
    num_seq_processed = 0  # Number of sequences processed
    N_tok = 0.0     # Total number of valid tokens

    # Process each sequence in the evaluation dataset
    for x_seq in data_set:
        log_seq_p = 0.0  # Log probability for current sequence
        
        # Create mask: 1 for valid tokens (>=0), 0 for padding (<0)
        mk = tf.cast(tf.greater_equal(x_seq, 0), dtype=tf.float32)
        
        # Step through each time position in the sequence
        for t in range(x_seq.shape[1]):
            # Extract token indices at time t for all sequences in batch
            i_t = np.expand_dims(x_seq[:, t], axis=1)  # Shape: [batch_size, 1]
            # Extract corresponding mask values
            m_t = tf.expand_dims(mk[:, t], axis=1)     # Shape: [batch_size, 1]

            # Convert token indices to one-hot encoding
            x_t = tf.squeeze(tf.one_hot(i_t, depth=vocab.size))  # Shape: [batch_size, vocab_size]
            # Handle single-sequence case (squeeze removes batch dimension)
            if i_t.shape[0] == 1:
                x_t = tf.expand_dims(x_t, axis=0)
            
            # Forward pass through P-TNCN model
            # is_eval=True: evaluation mode (no learning updates)
            # beta: temporal integration parameter for NCN dynamics
            # alpha: error-driven learning rate parameter
            x_logits, x_mu = model.forward(x_t, m_t, is_eval=True, beta=beta, alpha=alpha_e)

            # Only compute loss after warm-up period (t >= t_prime)
            # This allows model to build initial context before making predictions
            if t >= t_prime:
                if use_low_dim_eval is False:
                    # Standard evaluation: compute categorical cross-entropy
                    log_seq_p += fast_log_loss(x_mu, i_t)
                    
                    # Compute token-level accuracy
                    x_pred_t = tf.expand_dims(tf.cast(tf.argmax(x_mu, 1), dtype=tf.int32), axis=1)
                    comp = tf.cast(tf.equal(x_pred_t, i_t), dtype=tf.float32) * m_t
                    acc += tf.reduce_sum(comp)  # Accumulate correct predictions
                else:
                    # Alternative low-dimensional evaluation (rarely used)
                    log_seq_p += -tf.reduce_sum(tf.math.log(x_mu) * x_t)

                # Count valid tokens for normalization
                if normalize_by_num_seq is False:
                    N_tok += tf.reduce_sum(m_t)
        
        # Clear model's internal temporal state between sequences
        # Important for P-TNCN to reset between independent sequences
        model.clear_var_history()
        
        # Accumulate sequence-level metrics
        cost += log_seq_p
        if normalize_by_num_seq is True:
            N_tok += x_seq.shape[0]  # Count sequences instead of tokens

        num_seq_processed += x_seq.shape[0]

        # Progress display during evaluation
        N_S = N_tok
        print("\r >> Evaluated on {0} seq, {1} items - cost = {2}".format(
            num_seq_processed, N_tok, (cost / (N_S))), end="")
    
    print()  # New line after progress display
    
    # Normalize metrics by total count
    cost = cost / N_tok
    acc = acc / N_tok
    mse = mse / N_tok
    
    # Compute perplexity
    if calc_bpc is True:
        # Bits per character: convert from nats to bits
        ppl = cost * (1.0 / np.log(2.0))
    else:
        # Standard perplexity: exp(cross-entropy)
        ppl = tf.exp(cost)

    return cost, acc, ppl, mse

def eval_model_timed(model, train_data, dev_data, subtrain_data=None):
    """
    Wrapper for model evaluation that includes wall-clock timing.
    
    Args:
        model: P-TNCN model instance
        train_data: Training dataset (or subset for faster evaluation)
        dev_data: Validation/development dataset
        subtrain_data: Optional smaller training subset for faster evaluation
    
    Returns:
        cost_i: Training loss
        acc_i: Training accuracy  
        vcost_i: Validation loss
        vacc_i: Validation accuracy
        eval_time_v: Wall-clock evaluation time in seconds
        ppl_i: Training perplexity
        vppl_i: Validation perplexity
        mse_i: Training MSE (unused)
        vmse_i: Validation MSE (unused)
    
    Purpose: Provides both training and validation metrics with timing information
             for monitoring training progress and computational efficiency.
    """
    start_v = time.process_time()  # Start timing
    
    # Evaluate on training data (or subset if provided)
    if subtrain_data is not None:
        cost_i, acc_i, ppl_i, mse_i = eval_model(model, subtrain_data)
    else:
        cost_i, acc_i, ppl_i, mse_i = eval_model(model, train_data)
    
    # Evaluate on validation data
    vcost_i, vacc_i, vppl_i, vmse_i = eval_model(model, dev_data)
    
    end_v = time.process_time()    # End timing
    eval_time_v = end_v - start_v  # Calculate elapsed time
    
    return cost_i, acc_i, vcost_i, vacc_i, eval_time_v, ppl_i, vppl_i, mse_i, vmse_i

###########################################################################################################
# Configuration Parameters and Hyperparameters
###########################################################################################################

# Dataset file paths (Penn Treebank character-level data)
train_fname = "../data/ptb_char/trainX.txt"        # Main training sequences
subtrain_fname = "../data/ptb_char/subX.txt"       # Subset for faster training evaluation
dev_fname = "../data/ptb_char/validX.txt"          # Validation sequences
vocab = "../data/ptb_char/vocab.txt"               # Character vocabulary mapping
out_dir = "../outputs/run-2"                       # Output directory for models and logs

# Evaluation and output configuration
calc_bpc = True                    # Calculate bits-per-character (True) vs perplexity (False)
use_low_dim_eval = False          # Use alternative evaluation method (typically False)
accum_updates = False             # Accumulate gradients across time steps (False = update each step)
normalize_by_num_seq = False      # Normalize by sequence count vs token count (don't change)
out_fun = "softmax"               # Output layer activation function

# Training hyperparameters
mb = 50                           # Mini-batch size for training
v_mb = 100                        # Mini-batch size for validation (can be larger)
eval_iter = 2000                  # Evaluate model every N sequences
t_prime = 1                       # Skip first t_prime time steps in loss computation

# Model architecture and optimization parameters
model_form = "ptncn"              # Model type identifier
n_e = 50                          # Number of training epochs
opt_type = "nag"                  # Optimizer: "nag"=Nesterov, "momentum", "adam", "rmsprop", "sgd"
init_type = "normal"              # Parameter initialization method
alpha = 0.075                     # Learning rate
momentum = 0.95                   # Momentum coefficient for momentum-based optimizers
update_radius = 1.0               # Gradient clipping radius (clips gradient norms)
param_radius = -30                # Parameter clipping radius (disabled when negative)
w_decay = -0.0001                 # Weight decay coefficient (disabled when negative)

# P-TNCN specific architecture parameters
hid_dim = 1000                    # Hidden layer dimensions
wght_sd = 0.05                    # Standard deviation for weight initialization
err_wght_sd = 0.05                # Standard deviation for error weight initialization
beta = 0.1                        # NCN temporal integration parameter (controls memory)
gamma = 1                         # Update scaling factor for P-TNCN learning
act_fun = "tanh"                  # Activation function for hidden layers
alpha_e = 0.001                   # Error-driven learning rate (from IEEE paper)

# Model loading and evaluation options
load_model = False                # Load pre-trained model instead of training from scratch
model_fname = "model_best.pkl"    # Filename for model loading/saving
eval_only = False                 # Only evaluate loaded model (skip training)

###########################################################################################################
# Model and Optimizer Initialization
###########################################################################################################

# Create TensorFlow variables for dynamic learning rate and momentum adjustment
moment_v = tf.Variable(momentum)   # Momentum coefficient (can be modified during training)
alpha_v = tf.Variable(alpha)       # Learning rate (can be modified during training)

# Initialize optimizer based on specified type
# Using TF 1.x optimizers for compatibility with existing P-TNCN implementation
if opt_type == "nag":
    # Nesterov Accelerated Gradient: momentum + lookahead gradient computation
    optimizer = tf.compat.v1.train.MomentumOptimizer(
        learning_rate=alpha_v, momentum=moment_v, use_nesterov=True)
elif opt_type == "momentum":
    # Standard momentum: exponential moving average of gradients
    optimizer = tf.compat.v1.train.MomentumOptimizer(
        learning_rate=alpha_v, momentum=moment_v, use_nesterov=False)
elif opt_type == "adam":
    # Adam: adaptive learning rates with momentum-like behavior
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=alpha_v)
elif opt_type == "rmsprop":
    # RMSprop: adaptive learning rates based on recent gradient magnitudes
    optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=alpha_v)
else:
    # Standard stochastic gradient descent
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=alpha_v)

print(" > Creating vocab filter...")
# Load vocabulary mapping (character ↔ integer index)
vocab = Vocab(vocab)
out_dim = vocab.size  # Output layer size = vocabulary size

print(" > Vocab.size = ", vocab.size)
# Initialize data loaders for training and validation
train_data = DataLoader(train_fname, mb)        # Training data with batch size mb
subtrain_data = DataLoader(subtrain_fname, v_mb) # Training subset for faster evaluation
dev_data = DataLoader(dev_fname, v_mb)          # Validation data with batch size v_mb

# Model initialization: load pre-trained or create new
if load_model is True or eval_only is True:
    # Load pre-trained model from disk
    print(" >> Loading pre-trained model: {0}{1}".format(out_dir, model_fname))
    fd = open("{0}{1}".format(out_dir, model_fname), 'rb')
    model = pickle.load(fd)
    fd.close()
else:
    # Create new P-TNCN model with specified architecture
    in_dim = -1  # Auto-determined from data (leave as -1)
    model = PTNCN("ptncn", out_dim, hid_dim, 
                  wght_sd=wght_sd, err_wght_sd=err_wght_sd,
                  act_fun=act_fun, out_fun=out_fun, in_dim=in_dim)

# Display model complexity (number of parameters/synapses)
print(" Model.Complexity = {0} synapses".format(model.get_complexity()))

# Initial model evaluation before training
# Provides baseline performance metrics and validates model setup
cost_i, acc_i, vcost_i, vacc_i, eval_time_v, ppl_i, vppl_i, mse_i, vmse_i = eval_model_timed(
    model, train_data, dev_data, subtrain_data=subtrain_data)

print(" -1: Tr.L = {0} Tr.Acc = {1} V.L = {2} V.Acc = {3} Tr.MSE = {4} V.MSE = {5} in {6} s".format(
    cost_i, acc_i, vcost_i, vacc_i, mse_i, vmse_i, eval_time_v))

# Store initial validation loss for best model tracking
vcost_im1 = vcost_i

# Training and logging setup (skip if evaluation-only mode)
if eval_only is False:
    # Initialize performance log file
    log = open("{0}{1}".format(out_dir, "perf.txt"), "w")
    log.write("Iter, Loss, PPL, Acc, VLoss, VPPL, VAcc\n")  # CSV header
    log.flush()
    # Write initial evaluation results (iteration -1)
    log.write("{0}, {1}, {2}, {3}, {4}, {5}, {6}\n".format(
        -1, cost_i, ppl_i, acc_i, vcost_i, vppl_i, vacc_i))
    log.flush()

    # Initialize parameter norm tracking log
    norm_log = open("{0}{1}".format(out_dir, "norm_log.txt"), "w")
    norm_log.write("{0}\n".format(theta_norms_to_str(model, is_header=True)))  # Header
    norm_log.write("{0}\n".format(theta_norms_to_str(model)))                  # Initial norms
    norm_log.flush()

    ###########################################################################################################
    # Main Training Loop
    ###########################################################################################################
    
    # Training loop: iterate through epochs
    for e in range(n_e):
        # Initialize epoch-level counters
        num_seq_processed = 0  # Number of sequences processed in this epoch
        N_tok = 0.0           # Total number of tokens processed
        start = time.process_time()  # Start timing for this epoch
        
        ########################################################################
        # Inner Training Loop: Process sequences and time steps
        ########################################################################
        tick = 0  # Counter for sequences since last evaluation
        
        # Process each mini-batch of sequences
        for x_seq in train_data:
            # Create mask: 1 for valid tokens (>=0), 0 for padding (<0)
            mk = tf.cast(tf.greater_equal(x_seq, 0), dtype=tf.float32)
            delta_accum = []  # Accumulator for gradients (if accum_updates=True)
            
            # Process each time step in the sequence
            for t in range(x_seq.shape[1]):
                # Extract token indices at current time step
                i_t = np.expand_dims(x_seq[:, t], axis=1)  # Shape: [batch_size, 1]
                # Extract mask values at current time step
                m_t = tf.expand_dims(mk[:, t], axis=1)     # Shape: [batch_size, 1]

                # Convert token indices to one-hot encoding
                x_t = tf.squeeze(tf.one_hot(i_t, depth=vocab.size))  # Shape: [batch_size, vocab_size]
                # Handle single-sequence batch case
                if i_t.shape[0] == 1:
                    x_t = tf.expand_dims(x_t, axis=0)
                
                # Forward pass through P-TNCN model
                # is_eval=False: training mode (enables learning dynamics)
                # beta: temporal integration parameter for NCN
                # alpha_e: error-driven learning rate
                x_logits, x_mu = model.forward(x_t, m_t, is_eval=False, beta=beta, alpha=alpha_e)

                # Compute and apply updates (skip initial t_prime time steps)
                if t >= t_prime:
                    # Compute P-TNCN specific gradients/updates
                    # gamma: scaling factor for updates
                    # update_radius: gradient clipping radius
                    delta = model.compute_updates(m_t, gamma=gamma, update_radius=update_radius)
                    
                    # Normalize gradients by mini-batch size
                    N_mb = x_t.shape[0]
                    for p in range(len(delta)):
                        delta[p] = delta[p] * (1.0 / (N_mb * 1.0))

                    # Apply updates: either immediately or accumulate for later
                    if accum_updates is False:
                        # Immediate update: apply gradients at each time step
                        optimizer.apply_gradients(zip(delta, model.param_var))

                        # Optional weight decay regularization
                        if w_decay > 0.0:
                            for p in range(len(delta)):
                                delta_var = delta[p]
                                delta[p] = tf.subtract(delta_var, (delta_var * w_decay))

                        # Optional parameter clipping to prevent explosion
                        if param_radius > 0.0:
                            for p in range(len(model.param_var)):
                                old_var = model.param_var[p]
                                old_var.assign(tf.clip_by_norm(old_var, param_radius, axes=[1]))
                    else:
                        # Gradient accumulation: sum gradients across time steps
                        if len(delta_accum) > 0:
                            # Add to existing accumulated gradients
                            for p in range(len(delta)):
                                delta_accum[p] = tf.add(delta_accum[p], delta[p])
                        else:
                            # Initialize accumulator with first gradients
                            for p in range(len(delta)):
                                delta_accum.append(delta[p])

                # Count valid tokens for progress tracking
                N_tok += tf.reduce_sum(m_t)

            # Apply accumulated gradients if using accumulation mode
            if accum_updates is True:
                optimizer.apply_gradients(zip(delta_accum, model.param_var))

                # Parameter clipping after accumulated update
                if param_radius > 0.0:
                    for p in range(len(model.param_var)):
                        old_var = model.param_var[p]
                        old_var.assign(tf.clip_by_norm(old_var, param_radius, axes=[1]))
            
            # Clear P-TNCN internal temporal state between sequences
            # Essential for proper sequence-to-sequence independence
            model.clear_var_history()
            ########################################################################

            # Update progress counters
            num_seq_processed += x_seq.shape[0]
            tick += x_seq.shape[0]
            print("\r  >> Processed {0} seq, {1} tok ".format(num_seq_processed, N_tok), end="")

            # Periodic evaluation and model checkpointing
            if tick >= eval_iter:
                print()  # New line after progress display
                
                # Comprehensive evaluation on training and validation sets
                cost_i, acc_i, vcost_i, vacc_i, eval_time_v, ppl_i, vppl_i, mse_i, vmse_i = eval_model_timed(
                    model, train_data, dev_data, subtrain_data=subtrain_data)
                
                # Display current performance metrics
                print(" {0}: Tr.L = {1} Tr.Acc = {2} V.L = {3} V.Acc = {4} Tr.MSE = {5} V.MSE = {6} in {7} s".format(
                    e, cost_i, acc_i, vcost_i, vacc_i, mse_i, vmse_i, eval_time_v))
                
                # Log performance metrics to CSV file
                log.write("{0}, {1}, {2}, {3}, {4}, {5}, {6}\n".format(
                    e, cost_i, ppl_i, acc_i, vcost_i, vppl_i, vacc_i))
                log.flush()
                
                # Log parameter norms for monitoring training dynamics
                norm_log.write("{0}\n".format(theta_norms_to_str(model)))
                norm_log.flush()

                # Save current model checkpoint
                fd = open("{0}model{1}.pkl".format(out_dir, e), 'wb')
                pickle.dump(model, fd)
                fd.close()

                # Save best model if validation loss improved
                if vcost_i <= vcost_im1:
                    fd = open("{0}model_best.pkl".format(out_dir), 'wb')
                    pickle.dump(model, fd)
                    fd.close()
                    vcost_im1 = vcost_i  # Update best validation loss

                tick = 0  # Reset evaluation counter
        print()  # New line after sequence processing
        ########################################################################
        # End of Epoch Processing
        ########################################################################
        end = time.process_time()
        train_time = end - start
        print("  -> Trained time = {0} s".format(train_time))
        
        # Final evaluation if there were remaining sequences since last eval
        if tick > 0:
            # Evaluate model performance at end of epoch
            cost_i, acc_i, vcost_i, vacc_i, eval_time_v, ppl_i, vppl_i, mse_i, vmse_i = eval_model_timed(
                model, train_data, dev_data, subtrain_data=subtrain_data)
            
            # Display final epoch metrics
            print(" {0}: Tr.L = {1} Tr.Acc = {2} V.L = {3} V.Acc = {4} Tr.MSE = {5} V.MSE = {6} in {7} s".format(
                e, cost_i, acc_i, vcost_i, vacc_i, mse_i, vmse_i, eval_time_v))
            
            # Log final epoch performance
            log.write("{0}, {1}, {2}, {3}, {4}, {5}, {6}\n".format(
                e, cost_i, acc_i, ppl_i, vcost_i, vacc_i, vppl_i))
            log.flush()
            
            # Log final parameter norms
            norm_log.write("{0}\n".format(theta_norms_to_str(model)))
            norm_log.flush()

            # Save final epoch model
            fd = open("{0}model{1}.pkl".format(out_dir, e), 'wb')
            pickle.dump(model, fd)
            fd.close()

            # Update best model if validation loss improved
            if vcost_i <= vcost_im1:
                fd = open("{0}model_best.pkl".format(out_dir), 'wb')
                pickle.dump(model, fd)
                fd.close()
                vcost_im1 = vcost_i

            tick = 0

    # Close log files after training completion
    log.close()
    norm_log.close()
