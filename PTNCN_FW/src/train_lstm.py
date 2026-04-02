import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
"""
Trains an LSTM model for discrete token prediction, forecasting over sequences
of one-hot encoded integer symbols.

@author: Ankur Mali
"""
import time
import sys
import pickle
import math

sys.path.insert(0, 'models/')
sys.path.insert(0, 'utils/')

import tensorflow as tf
import numpy as np
from lstm import LSTM
from data import Vocab
from seq_sampler import DataLoader

seed = 1234
tf.random.set_seed(seed=seed)
np.random.seed(seed)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("GPU Device Name: ", tf.test.gpu_device_name())

###########################################################################################################
# define helper functions first....
###########################################################################################################

def seq_to_tokens(idx_seq, vocab, mb_idx=0):
    tok_seq = ""
    for i in range(0, idx_seq.shape[0]):
        idx = idx_seq[i][mb_idx].numpy()
        tok_seq += vocab.idx2token(idx) + " "
    return tok_seq

def theta_norms_to_str(model, is_header=False):
    str = ""
    theta = model.collect_params()
    for param_name in theta:
        if is_header is False:
            str += "{0}".format(tf.norm(theta[param_name],ord="euclidean"))
        else:
            str += "{0}".format(param_name)
        str += ","
    str = str[:-1]
    return str

def create_fixed_point(data_set, n_rounds=1, n_seq_total=-1):
    samp_seq_list = []
    n_seq = 0
    if n_seq_total > 0:
        flag = True
        while flag:
            for tr, ip, sg, mk, nxt_snt_flag in data_set:
                samp_seq_list.append( (tr,ip,sg,mk) )
                n_seq += 1
                if n_seq >= n_seq_total:
                    flag = False
                    break
    else:
        debug_n_rounds = -1
        for r in range(n_rounds):
            n_seq = 0
            for tr, ip, sg, mk, nxt_snt_flag in data_set:
                if debug_n_rounds <= 0:
                    samp_seq_list.append( (tr,ip,sg,mk) )
                else:
                    if r < debug_n_rounds:
                        samp_seq_list.append( (tr,ip,sg,mk) )
    return samp_seq_list

def fast_log_loss(probs, y_ind):
    loss = 0.0
    py = probs.numpy()
    for i in range(0, y_ind.shape[0]):
        ti = y_ind[i][0]
        if ti >= 0:
            py = probs[i,ti]
            if py <= 0.0:
                py = 1e-8
            loss += np.log(py)
    return -loss

def eval_model(model, data_set, debug_step_print=False):
    cost = 0.0
    acc = 0.0
    mse = 0.0
    num_seq_processed = 0
    N_tok = 0.0

    for x_seq in data_set:
        log_seq_p = 0.0
        mk = tf.cast(tf.greater_equal(x_seq, 0), dtype=tf.float32)
        for t in range(x_seq.shape[1]):
            i_t = np.expand_dims(x_seq[:,t],axis=1)
            m_t = tf.expand_dims(mk[:,t],axis=1)

            x_t = tf.squeeze( tf.one_hot(i_t,depth=vocab.size) )
            if i_t.shape[0] == 1:
                x_t = tf.expand_dims(x_t, axis=0)
            y = model.forward(x_t, m_t)

            if t >= t_prime:
                if use_low_dim_eval is False:
                    log_seq_p += fast_log_loss(y, i_t)
                    x_pred_t = tf.expand_dims(tf.cast(tf.argmax(y,1),dtype=tf.int32),axis=1)
                    comp = tf.cast(tf.equal(x_pred_t, i_t),dtype=tf.float32) * m_t
                    acc += tf.reduce_sum( comp )
                else:
                    log_seq_p += -tf.reduce_sum( tf.math.log(y) * x_t )

                if normalize_by_num_seq is False:
                    N_tok += tf.reduce_sum(m_t)
        model.clear_var_history()
        cost += log_seq_p
        if normalize_by_num_seq is True:
            N_tok += x_seq.shape[0]

        num_seq_processed += x_seq.shape[0]

        N_S = N_tok
        print("\r >> Evaluated on {0} seq, {1} items - cost = {2}".format(num_seq_processed, N_tok, (cost /(N_S)) ), end="")
    print()
    cost = cost / N_tok
    acc = acc / N_tok
    mse = mse / N_tok
    if calc_bpc is True:
        ppl = cost * (1.0 / np.log(2.0))
    else:
        ppl = tf.exp(cost)

    return cost, acc, ppl, mse

def eval_model_timed(model, train_data, dev_data, subtrain_data=None):
    start_v = time.process_time()
    if subtrain_data is not None:
        cost_i, acc_i, ppl_i, mse_i = eval_model(model, subtrain_data)
    else:
        cost_i, acc_i, ppl_i, mse_i = eval_model(model, train_data)
    vcost_i, vacc_i, vppl_i, vmse_i = eval_model(model, dev_data)
    end_v = time.process_time()
    eval_time_v = end_v - start_v
    return cost_i, acc_i, vcost_i, vacc_i, eval_time_v, ppl_i, vppl_i, mse_i, vmse_i

###########################################################################################################
# set simulation meta-parameters
###########################################################################################################

train_fname = "../data/ptb_char/trainX.txt"
subtrain_fname = "../data/ptb_char/subX.txt"
dev_fname = "../data/ptb_char/validX.txt"
vocab = "../data/ptb_char/vocab.txt"
out_dir = "./out_dir_lstm/"
calc_bpc = True
use_low_dim_eval = False
normalize_by_num_seq = False
out_fun = "softmax"

# training meta-parameters
mb = 200
v_mb = 400
eval_iter = float('inf')
t_prime = 1

# meta-parameters for model itself
model_form = "lstm"
n_e = 30
opt_type = "adam"
init_type = "normal"
learning_rate = 0.1
momentum = 0.9
w_decay = 0.0001
hid_dim = 250
wght_sd = 0.05

load_model = False
model_fname = "model_best.pkl"
eval_only = False

###########################################################################################################
# initialize the program
###########################################################################################################

# Set up optimizer
if opt_type == "adam":
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
elif opt_type == "sgd":
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
elif opt_type == "rmsprop":
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
else:
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

print(" > Creating vocab filter...")
vocab = Vocab(vocab)
out_dim = vocab.size

print(" > Vocab.size = ", vocab.size)
train_data = DataLoader(train_fname, mb)
subtrain_data = DataLoader(subtrain_fname, v_mb)
dev_data = DataLoader(dev_fname, v_mb)

if load_model is True or eval_only is True:
    print(" >> Loading pre-trained model: {0}{1}".format(out_dir,model_fname))
    fd = open("{0}{1}".format(out_dir, model_fname), 'rb')
    model = pickle.load(fd)
    fd.close()
else:
    model = LSTM("lstm", out_dim, hid_dim, wght_sd=wght_sd, init_type=init_type)

print(" Model.Complexity = {0} synapses".format(model.get_complexity()))

cost_i, acc_i, vcost_i, vacc_i, eval_time_v, ppl_i, vppl_i, mse_i, vmse_i = eval_model_timed(model, train_data, dev_data, subtrain_data=subtrain_data)
print(" -1: Tr.L = {0} Tr.Acc = {1} V.L = {2} V.Acc = {3} Tr.MSE = {4} V.MSE = {5} in {6} s".format(cost_i, acc_i, vcost_i, vacc_i, mse_i, vmse_i, eval_time_v))
vcost_im1 = vcost_i

if eval_only is False:
    log = open("{0}{1}".format(out_dir,"perf.txt"),"w")
    log.write("Iter, Loss, PPL, Acc, VLoss, VPPL, VAcc\n")
    log.flush()
    log.write("{0}, {1}, {2}, {3}, {4}, {5}, {6}\n".format(-1, cost_i, ppl_i, acc_i, vcost_i, vppl_i, vacc_i))
    log.flush()

    norm_log = open("{0}{1}".format(out_dir,"norm_log.txt"),"w")
    norm_log.write("{0}\n".format(theta_norms_to_str(model,is_header=True)))
    norm_log.write("{0}\n".format(theta_norms_to_str(model)))
    norm_log.flush()

    for e in range(n_e):
        num_seq_processed = 0
        N_tok = 0.0
        start = time.process_time()
        
        # Training loop
        tick = 0
        for x_seq in train_data:
            mk = tf.cast(tf.greater_equal(x_seq, 0), dtype=tf.float32)
            
            for t in range(x_seq.shape[1]):
                i_t = np.expand_dims(x_seq[:,t],axis=1)
                m_t = tf.expand_dims(mk[:,t],axis=1)

                x_t = tf.squeeze( tf.one_hot(i_t,depth=vocab.size) )
                if i_t.shape[0] == 1:
                    x_t = tf.expand_dims(x_t, axis=0)
                
                with tf.GradientTape() as tape:
                    y = model.forward(x_t, m_t)
                    if t >= t_prime:
                        # Get valid indices (where mask is 1)
                        valid_indices = tf.where(m_t[:, 0] > 0)
                        if tf.size(valid_indices) > 0:
                            # Get valid labels and predictions
                            valid_labels = tf.gather_nd(i_t, valid_indices)
                            valid_predictions = tf.gather_nd(y, valid_indices)
                            
                            # Calculate loss only for valid predictions
                            loss = tf.reduce_mean(
                                tf.keras.losses.sparse_categorical_crossentropy(
                                    valid_labels, 
                                    valid_predictions,
                                    from_logits=True
                                )
                            )
                            
                            if w_decay > 0.0:
                                l2_loss = sum(tf.nn.l2_loss(w) for w in model.param_var)
                                loss += w_decay * l2_loss
                        else:
                            loss = 0.0
                
                if t >= t_prime and loss != 0.0:
                    grads = tape.gradient(loss, model.param_var)
                    optimizer.apply_gradients(zip(grads, model.param_var))
                    N_tok += tf.reduce_sum(m_t)

            model.clear_var_history()
            num_seq_processed += x_seq.shape[0]
            tick += x_seq.shape[0]
            print("\r  >> Processed {0} seq, {1} tok ".format(num_seq_processed, N_tok), end="")

            if tick >= eval_iter:
                print()
                cost_i, acc_i, vcost_i, vacc_i, eval_time_v, ppl_i, vppl_i, mse_i, vmse_i = eval_model_timed(model, train_data, dev_data, subtrain_data=subtrain_data)
                print(" {0}: Tr.L = {1} Tr.Acc = {2} V.L = {3} V.Acc = {4} Tr.MSE = {5} V.MSE = {6} in {7} s".format(e, cost_i, acc_i, vcost_i, vacc_i, mse_i, vmse_i, eval_time_v))
                log.write("{0}, {1}, {2}, {3}, {4}, {5}, {6}\n".format(e, cost_i, ppl_i, acc_i, vcost_i, vppl_i, vacc_i))
                log.flush()
                norm_log.write("{0}\n".format(theta_norms_to_str(model)))
                norm_log.flush()

                # save model to disk
                fd = open("{0}model{1}.pkl".format(out_dir,e), 'wb')
                pickle.dump(model, fd)
                fd.close()

                if vcost_i <= vcost_im1:
                    fd = open("{0}model_best.pkl".format(out_dir), 'wb')
                    pickle.dump(model, fd)
                    fd.close()
                    vcost_im1 = vcost_i

                tick = 0
        print()
        
        end = time.process_time()
        train_time = end - start
        print("  -> Trained time = {0} s".format(train_time))
        
        if tick > 0:
            cost_i, acc_i, vcost_i, vacc_i, eval_time_v, ppl_i, vppl_i, mse_i, vmse_i = eval_model_timed(model, train_data, dev_data, subtrain_data=subtrain_data)
            print(" {0}: Tr.L = {1} Tr.Acc = {2} V.L = {3} V.Acc = {4} Tr.MSE = {5} V.MSE = {6} in {7} s".format(e, cost_i, acc_i, vcost_i, vacc_i, mse_i, vmse_i, eval_time_v))
            log.write("{0}, {1}, {2}, {3}, {4}, {5}, {6}\n".format(e, cost_i, ppl_i, acc_i, vcost_i, vppl_i, vacc_i))
            log.flush()
            norm_log.write("{0}\n".format(theta_norms_to_str(model)))
            norm_log.flush()

            # save model to disk
            fd = open("{0}model{1}.pkl".format(out_dir,e), 'wb')
            pickle.dump(model, fd)
            fd.close()

            if vcost_i <= vcost_im1:
                fd = open("{0}model_best.pkl".format(out_dir), 'wb')
                pickle.dump(model, fd)
                fd.close()
                vcost_im1 = vcost_i

            tick = 0

    log.close()
