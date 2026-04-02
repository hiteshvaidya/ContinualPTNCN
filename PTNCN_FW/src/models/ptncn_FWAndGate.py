import tensorflow as tf
import sys
from utils import gelu1, create_dropout_mask, softmax, init_weights, gte, ltanh, d_relu6, standardize
import numpy as np

seed = 1234
tf.random.set_seed(seed=seed)
np.random.seed(seed)

'''
    Implementation of a 2-latent variable layer Parallel Temporal Neural Coding Network (P-TNCN), Ororbia and Mali 2019 IEEE TNNLS

    @author: Ankur Mali
'''
class PTNCN:
    def __init__(self, name, x_dim, hid_dim, wght_sd=0.025, err_wght_sd=0.025,act_fun="tanh", init_type="normal",
                 out_fun="identity",in_dim=-1,zeta=1.0): # constructor
        self.name = name
        self.act_fun = act_fun
        self.hid_dim = hid_dim
        self.x_dim = x_dim
        self.L1 = 0 #0.002 # lateral neuron penalty to encourage sparsity in representations
        self.init_type = init_type
        self.standardize = False # un-tested
        self.use_temporal_error_rule = False # Note: works better w/o temporal error rule in discrete-valued input space
        self.zeta = zeta # leave zeta = 1

        # variables that need to maintain state / long-running statistics over sequences
        self.zero_pad = tf.zeros([1,hid_dim])
        self.zf0 = None
        self.zf1 = None
        self.zf2 = None
        self.zf3 = None
        self.zf4 = None
        self.zf0_tm1 = None
        self.zf1_tm1 = None
        self.zf2_tm1 = None
        self.zf3_tm1 = None
        self.zf4_tm1 = None
        self.z1 = None
        self.z2 = None
        self.z3 = None
        self.z4 = None
        self.y1 = None
        self.y2 = None
        self.y3 = None
        self.y4 = None
        self.ex = None
        self.e1 = None
        self.e2 = None
        self.e3 = None
        self.e1v = None
        self.e2v = None
        self.e3v = None
        self.e4v = None
        self.e1v_tm1 = None
        self.e2v_tm1 = None
        self.e3v_tm1 = None
        self.e4v_tm1 = None
        self.x = None
        self.z_pad = None
        self.x_pad = None
        self.x_tm1 = None

        if in_dim <= 0:
            in_dim = x_dim
        self.in_dim = in_dim

        if self.zeta > 0.0:
            # top-down control weights
            self.U1 = tf.Variable( init_weights(self.init_type, [self.hid_dim, self.hid_dim], stddev=wght_sd, seed=seed) )
        # bottom-up data-driving weights
        self.M2 = tf.Variable( init_weights(self.init_type, [self.hid_dim, self.hid_dim], stddev=wght_sd, seed=seed) )
        self.M1 = tf.Variable( init_weights(self.init_type, [in_dim, self.hid_dim], stddev=wght_sd, seed=seed) )

        # top-down prediction weights
        self.W2 = tf.Variable( init_weights(self.init_type, [self.hid_dim, self.hid_dim], stddev=wght_sd, seed=seed) )
        self.W1 = tf.Variable( init_weights(self.init_type, [self.hid_dim, self.x_dim], stddev=wght_sd, seed=seed) )

        # recurrent memory weights
        self.V2 = tf.Variable( init_weights(self.init_type, [self.hid_dim, self.hid_dim], stddev=wght_sd, seed=seed) )
        self.V1 = tf.Variable( init_weights(self.init_type, [self.hid_dim, self.hid_dim], stddev=wght_sd, seed=seed) )

        # bottom-up error weights
        self.E2 = tf.Variable( init_weights(self.init_type, [self.hid_dim, self.hid_dim], stddev=err_wght_sd, seed=seed) )
        self.E1 = tf.Variable( init_weights(self.init_type, [self.x_dim, self.hid_dim], stddev=err_wght_sd, seed=seed) )

        # set up parameter variables for the model
        self.param_var = [] # book-keeping needed for using TF's weight update rules (or "optimizers")
        self.param_var.append(self.W1)
        self.param_var.append(self.E1)
        self.param_var.append(self.W2)
        self.param_var.append(self.E2)
        if self.zeta > 0.0:
            self.param_var.append(self.U1)
        self.param_var.append(self.M1)
        self.param_var.append(self.M2)
        self.param_var.append(self.V1)
        self.param_var.append(self.V2)

        # initialize the activation function
        self.act_fx = None
        if self.act_fun is "gelu":
            self.act_fx = gelu1
        elif self.act_fun is "relu6":
            self.act_fx = tf.nn.relu6
        elif self.act_fun is "relu":
            self.act_fx = tf.nn.relu
        elif self.act_fun is "sigmoid":
            self.act_fx = tf.nn.sigmoid
        elif self.act_fun is "sign":
            self.act_fx = tf.sign
        elif self.act_fun is "tanh":
            self.act_fx = tf.nn.tanh
        elif self.act_fun is "ltanh":
            self.act_fx = ltanh
        else:
            self.act_fx = gte

        self.out_fx = tf.identity
        if out_fun is "softmax":
            self.out_fx = softmax
        elif out_fun is "tanh":
            self.out_fx = tf.nn.tanh
        elif out_fun is "sigmoid":
            self.out_fx = tf.nn.sigmoid

    def act_dx(self, h, z):
        """
            Hasty derivative function handler
        """
        if self.act_fun == "tanh": # d/dh = 1 - tanh^2(h)
            return -(z * z) + 1.0
        elif self.act_fun == "ltanh": # d/dh = 1 - tanh^2(h)
            return d_ltanh(z)
        elif self.act_fun == "relu6":
            return d_relu6(h)
        elif self.act_fun == "relu":
            return d_relu(h)
        else:
            print("ERROR: deriv/dx fun not specified:{0}".format(self.act_fun ))
            sys.exit(0)

    def collect_params(self):
        """
            Routine for collecting all synaptic weight matrix/vectors from
            transformer model in a named dictionary (for norm printing)
        """
        theta = dict()
        theta["W1"] = self.W1
        theta["W2"] = self.W2
        theta["E1"] = self.E1
        theta["E2"] = self.E2
        theta["M1"] = self.M1
        theta["M2"] = self.M2
        if self.zeta > 0.0:
            theta["U1"] = self.U1
        theta["V1"] = self.V1
        theta["V2"] = self.V2
        return theta

    def get_complexity(self):
        """
            Routine for measuring model complexity in terms of number of synaptic weight parameters
        """
        wght_cnt = 0
        for i in range(len(self.param_var)):
            wr = self.param_var[i].shape[0]
            wc = self.param_var[i].shape[1]
            wght_cnt += (wr * wc)
        return wght_cnt

    def forward(self, x , m, K=5, beta=0.2, alpha=1, is_eval=True):
        """
            Forward inference step function for P-TNCN (note this maintains state)
        """
        y_logits = None
        y_mu = None
        x_logits = None
        x_mu = None
        x_ = tf.cast(x, dtype=tf.float32)

        if self.zf1 is not None:
            self.zf0_tm1 = self.zf0
            if self.x_tm1 is not None:
                self.zf0_tm1 = self.x_tm1
            self.zf1_tm1 = self.y1
            self.zf2_tm1 = self.y2
            self.e1v_tm1 = self.e1v
            self.e2v_tm1 = self.e2v
        else:
            if self.z_pad is None:
                self.z_pad = tf.zeros([x.shape[0], self.hid_dim]) #tf.tile(self.zero_pad, [x.shape[0],1])
                self.x_pad = tf.zeros([x.shape[0], self.in_dim])
            else:
                if self.z_pad.shape[0] != x.shape[0]:
                    self.z_pad = tf.zeros([x.shape[0], self.hid_dim])
                    self.x_pad = tf.zeros([x.shape[0], self.in_dim])
            self.zf0_tm1 = self.x_pad #x_ * 0
            self.zf1_tm1 = self.z_pad
            self.zf2_tm1 = self.z_pad
            self.e1v_tm1 = self.z_pad
            self.e2v_tm1 = self.z_pad
            self.z1 = self.z_pad
            self.z2 = self.z_pad
            self.z3 = self.z_pad
            self.e1 = self.z_pad
            self.e2 = self.z_pad
            self.ex = self.zf0_tm1

        # init states of variable at layer 0
        self.zf0 = x_ #tf.cast(tf.greater(x_, 0.0), dtype=tf.float32)

        # compute new states
        if self.zeta > 0.0:
            self.z2 = tf.add(tf.matmul(self.zf1_tm1, self.M2), tf.matmul(self.zf2_tm1, self.V2))
        else:
            self.z2 = tf.matmul(self.zf2_tm1, self.V2)
        #self.z2 = tf.add(tf.matmul(self.zf1_tm1, self.M2), tf.matmul(self.zf2_tm1, self.V2))
        if self.standardize is True:
            self.z2 = standardize( self.z2 )
        self.zf2 = self.act_fx(self.z2)
        z1_mu = tf.matmul(self.zf2, self.W2)

        if self.zeta > 0.0:
            self.z1 = tf.add(tf.add(tf.matmul(self.zf0_tm1, self.M1), tf.matmul(self.zf2_tm1, self.U1)), tf.matmul(self.zf1_tm1, self.V1))
        else:
            self.z1 = tf.add(tf.matmul(self.zf0_tm1, self.M1), tf.matmul(self.zf1_tm1, self.V1))
        if self.standardize is True:
            self.z1 = standardize( self.z1 )
        self.zf1 = self.act_fx(self.z1)
        x_logits = tf.matmul(self.zf1, self.W1)
        x_mu = self.out_fx( x_logits ) #tf.nn.sigmoid( x_logits )

        # compute local errors and state perturbations
        self.e1 = tf.subtract(z1_mu, self.zf1) * m
        self.ex = tf.subtract(x_mu, self.zf0) * m
        d2 = tf.matmul(self.e1, self.E2)
        d1 = tf.subtract(tf.matmul(self.ex, self.E1), self.e1 * alpha)
        if self.L1 > 0.0:# inject the weak lateral sparsity prior here
            d2 = tf.add(d2, tf.sign(self.z2) * self.L1)
            d1 = tf.add(d1, tf.sign(self.z1) * self.L1)
        # compute state targets
        self.y2 = self.act_fx( tf.subtract(self.z2, d2 * beta) )
        self.y1 = self.act_fx( tf.subtract(self.z1, d1 * beta) )
        # compute temporal weight corrections
        self.e2v = tf.subtract(self.zf2, self.y2) * m
        self.e1v = tf.subtract(self.zf1, self.y1) * m

        self.e1v_tm1 = self.e1v
        self.e2v_tm1 = self.e2v

        '''
        # no need to waste the multiplications to mask these out as only the error neurons drive weight updates in this case
        self.y1 = self.y1 * m
        self.y2 = self.y2 * m
        self.zf1 = self.zf1 * m
        self.zf2 = self.zf2 * m
        '''

        return x_logits, x_mu

    def compute_updates(self, m, gamma=1.0, update_radius=-1.0):
        """
            Computes the perturbations needed to adjust P-TNCN's synaptic weight parameters
        """
        delta_list = []
        # W1
        dW = tf.matmul(self.zf1, self.ex, transpose_a=True)
        if update_radius > 0.0:
            dW = tf.clip_by_norm(dW, update_radius)
        delta_list.append(dW )

        # E1
        if self.use_temporal_error_rule is True:
            dW = None
            dW = tf.matmul(self.ex, tf.subtract(self.e1v, self.e1v_tm1), transpose_a=True) * -gamma
        else:
            dW = tf.transpose(dW) * gamma
        if update_radius > 0.0:
            dW = tf.clip_by_norm(dW, update_radius)
        delta_list.append(dW )

        # W2
        dW = None
        dW = tf.matmul(self.zf2, self.e1v, transpose_a=True)
        if update_radius > 0.0:
            dW = tf.clip_by_norm(dW, update_radius)
        delta_list.append(dW )

        # E2
        if self.use_temporal_error_rule is True:
            dW = None
            dW = tf.matmul(self.e1, tf.subtract(self.e2v, self.e2v_tm1), transpose_a=True) * -gamma
        else:
            dW = tf.transpose(dW) * gamma
        if update_radius > 0.0:
            dW = tf.clip_by_norm(dW, update_radius)
        delta_list.append(dW )

        if self.zeta > 0.0:
            # U1
            dW = None
            dW = tf.matmul(self.zf2_tm1, self.e1v, transpose_a=True)
            if update_radius > 0.0:
                dW = tf.clip_by_norm(dW, update_radius)
            delta_list.append(dW )

        # M1
        dW = None
        dW = tf.matmul(self.zf0_tm1, self.e1v, transpose_a=True)
        if update_radius > 0.0:
            dW = tf.clip_by_norm(dW, update_radius)
        delta_list.append(dW )
        
        # M2
        dW = None
        dW = tf.matmul(self.zf1_tm1, self.e2v, transpose_a=True)
        if update_radius > 0.0:
            dW = tf.clip_by_norm(dW, update_radius)
        delta_list.append(dW )

        # V1
        dW = None
        dW = tf.matmul(self.zf1_tm1, self.e1v, transpose_a=True)
        if update_radius > 0.0:
            dW = tf.clip_by_norm(dW, update_radius)
        delta_list.append(dW )

        # V2
        dW = None
        dW = tf.matmul(self.zf2_tm1, self.e2v, transpose_a=True)
        if update_radius > 0.0:
            dW = tf.clip_by_norm(dW, update_radius)
        delta_list.append(dW )

        return delta_list

    def clear_var_history(self):
        """
            Variable-clearing function -- helps to reset P-TNCN state to NULL for new disjoint sequences,
            otherwise, the model will continue to update its stateful dynamical variables
        """
        self.zf0 = None
        self.zf1 = None
        self.zf2 = None
        self.zf3 = None
        self.zf4 = None
        self.zf0_tm1 = None
        self.zf1_tm1 = None
        self.zf2_tm1 = None
        self.zf3_tm1 = None
        self.zf4_tm1 = None
        self.z1 = None
        self.z2 = None
        self.z3 = None
        self.z4 = None
        self.y1 = None
        self.y2 = None
        self.y3 = None
        self.y4 = None
        self.ex = None
        self.e1 = None
        self.e2 = None
        self.e3 = None
        self.e1v = None
        self.e2v = None
        self.e3v = None
        self.e4v = None
        self.e1v_tm1 = None
        self.e2v_tm1 = None
        self.e3v_tm1 = None
        self.e4v_tm1 = None
        self.x = None

class PTNCN_Gate_FastWeight(PTNCN):
    def __init__(self, name, x_dim, hid_dim, wght_sd=0.025, err_wght_sd=0.025, act_fun="tanh", init_type="normal",
                 out_fun="identity", in_dim=-1, zeta=1.0, fast_weight_lr=0.01, fast_weight_decay=0.5):
        super().__init__(name, x_dim, hid_dim, wght_sd, err_wght_sd, act_fun, init_type, out_fun, in_dim, zeta)
        
        # Fast weight learning parameters - adjusted for stability
        self.fast_weight_lr = fast_weight_lr * 0.1  # Reduced learning rate
        self.fast_weight_decay = fast_weight_decay * 0.9  # Increased decay
        self.fast_weight_max_norm = 1.0  # Maximum norm for fast weights
        
        # Initialize fast weight matrices with smaller values
        self.A1 = tf.Variable(tf.random.normal([self.hid_dim, self.hid_dim], stddev=0.01), trainable=False)
        self.A2 = tf.Variable(tf.random.normal([self.hid_dim, self.hid_dim], stddev=0.01), trainable=False)
        
        # Initialize gate matrices with adjusted standard deviations
        # G1_1 and G2_1: maps from input to hidden (smaller stddev)
        self.G1_1 = tf.Variable(init_weights(self.init_type, [self.x_dim, self.hid_dim], stddev=wght_sd*0.5, seed=seed))
        self.G2_1 = tf.Variable(init_weights(self.init_type, [self.x_dim, self.hid_dim], stddev=wght_sd*0.5, seed=seed))
        # G1_2 and G2_2: maps from hidden to hidden (larger stddev)
        self.G1_2 = tf.Variable(init_weights(self.init_type, [self.hid_dim, self.hid_dim], stddev=wght_sd*1.5, seed=seed))
        self.G2_2 = tf.Variable(init_weights(self.init_type, [self.hid_dim, self.hid_dim], stddev=wght_sd*1.5, seed=seed))
        
        # Add new parameters to param_var list
        self.param_var.extend([self.G1_1, self.G1_2, self.G2_1, self.G2_2])
        
        # Initialize gate states
        self.g1 = None
        self.g2 = None

    def _update_fast_weights(self):
        """
        Update fast weights according to the paper:
        ΔA_2^t = η z^{t-1}_2(z^t_2)^T
        A_2^t = λ A^{t-1}_2 + η z^{t-1}_2(z^t_2)^T
        """
        # Update A2 with gradient clipping
        dA2 = tf.matmul(self.zf2_tm1, self.zf2, transpose_a=True)
        dA2 = tf.clip_by_norm(dA2, self.fast_weight_max_norm)  # Clip gradient
        dA2 = dA2 * self.fast_weight_lr
        new_A2 = self.A2 * self.fast_weight_decay + dA2
        # Apply norm constraint
        new_A2 = tf.clip_by_norm(new_A2, self.fast_weight_max_norm)
        self.A2.assign(new_A2)
        
        # Update A1 with gradient clipping
        dA1 = tf.matmul(self.zf1_tm1, self.zf1, transpose_a=True)
        dA1 = tf.clip_by_norm(dA1, self.fast_weight_max_norm)  # Clip gradient
        dA1 = dA1 * self.fast_weight_lr
        new_A1 = self.A1 * self.fast_weight_decay + dA1
        # Apply norm constraint
        new_A1 = tf.clip_by_norm(new_A1, self.fast_weight_max_norm)
        self.A1.assign(new_A1)

    def compute_updates(self, m, gamma=1.0, update_radius=-1.0):
        """
        Computes the perturbations needed to adjust P-TNCN's synaptic weight parameters
        including gate weights
        """
        delta_list = []
        
        # W1
        dW = tf.matmul(self.zf1, self.ex, transpose_a=True)
        if update_radius > 0.0:
            dW = tf.clip_by_norm(dW, update_radius)
        delta_list.append(dW)

        # E1
        if self.use_temporal_error_rule is True:
            dW = tf.matmul(self.ex, tf.subtract(self.e1v, self.e1v_tm1), transpose_a=True) * -gamma
        else:
            dW = tf.transpose(dW) * gamma
        if update_radius > 0.0:
            dW = tf.clip_by_norm(dW, update_radius)
        delta_list.append(dW)

        # W2
        dW = tf.matmul(self.zf2, self.e1v, transpose_a=True)
        if update_radius > 0.0:
            dW = tf.clip_by_norm(dW, update_radius)
        delta_list.append(dW)

        # E2
        if self.use_temporal_error_rule is True:
            dW = tf.matmul(self.e1, tf.subtract(self.e2v, self.e2v_tm1), transpose_a=True) * -gamma
        else:
            dW = tf.transpose(dW) * gamma
        if update_radius > 0.0:
            dW = tf.clip_by_norm(dW, update_radius)
        delta_list.append(dW)

        if self.zeta > 0.0:
            # U1
            dW = tf.matmul(self.zf2_tm1, self.e1v, transpose_a=True)
            if update_radius > 0.0:
                dW = tf.clip_by_norm(dW, update_radius)
            delta_list.append(dW)

        # M1
        dW = tf.matmul(self.zf0_tm1, self.e1v, transpose_a=True)
        if update_radius > 0.0:
            dW = tf.clip_by_norm(dW, update_radius)
        delta_list.append(dW)

        # M2
        dW = tf.matmul(self.zf1_tm1, self.e2v, transpose_a=True)
        if update_radius > 0.0:
            dW = tf.clip_by_norm(dW, update_radius)
        delta_list.append(dW)

        # V1
        dW = tf.matmul(self.zf1_tm1, self.e1v, transpose_a=True)
        if update_radius > 0.0:
            dW = tf.clip_by_norm(dW, update_radius)
        delta_list.append(dW)

        # V2
        dW = tf.matmul(self.zf2_tm1, self.e2v, transpose_a=True)
        if update_radius > 0.0:
            dW = tf.clip_by_norm(dW, update_radius)
        delta_list.append(dW)

        # Gate weight updates
        # G1_1: Update based on input to hidden
        dG1_1 = tf.matmul(self.zf0, self.e1v, transpose_a=True) * gamma
        if update_radius > 0.0:
            dG1_1 = tf.clip_by_norm(dG1_1, update_radius)
        delta_list.append(dG1_1)

        # G1_2: Update based on hidden to hidden
        dG1_2 = tf.matmul(self.zf1_tm1, self.e1v, transpose_a=True) * gamma
        if update_radius > 0.0:
            dG1_2 = tf.clip_by_norm(dG1_2, update_radius)
        delta_list.append(dG1_2)

        # G2_1: Update based on input to hidden
        dG2_1 = tf.matmul(self.zf0, self.e2v, transpose_a=True) * gamma
        if update_radius > 0.0:
            dG2_1 = tf.clip_by_norm(dG2_1, update_radius)
        delta_list.append(dG2_1)

        # G2_2: Update based on hidden to hidden
        dG2_2 = tf.matmul(self.zf2_tm1, self.e2v, transpose_a=True) * gamma
        if update_radius > 0.0:
            dG2_2 = tf.clip_by_norm(dG2_2, update_radius)
        delta_list.append(dG2_2)

        return delta_list

    def forward(self, x, m, K=5, beta=0.2, alpha=1, is_eval=True):
        """
        Forward inference step function for P-TNCN with gates and fast weights
        Following the equations in the paper:
        g_l^t = σ(G_l^1 y_0^t + G_2^1 y_l^{t-1})
        a_l^t = g_l^t ⊙ (V_l y_0^t + U_l y_l^{t-1} + A_l^t y_l^{t-1})
        z_l^t = φ_z^l(a_l^t)
        """
        # Initialize states if needed
        if self.zf1 is None:
            super().forward(x, m, K, beta, alpha, is_eval)
            return None, None
            
        x_ = tf.cast(x, dtype=tf.float32)
        # print("Input shape (x_):", x_.shape)
        
        # Update previous states
        self.zf0_tm1 = self.zf0
        if self.x_tm1 is not None:
            self.zf0_tm1 = self.x_tm1
        self.zf1_tm1 = self.y1
        self.zf2_tm1 = self.y2
        self.e1v_tm1 = self.e1v
        self.e2v_tm1 = self.e2v
        
        # print("zf0_tm1 shape:", self.zf0_tm1.shape)
        # print("zf1_tm1 shape:", self.zf1_tm1.shape)
        # print("zf2_tm1 shape:", self.zf2_tm1.shape)
        
        # Initialize states if needed
        if self.z_pad is None or self.z_pad.shape[0] != x.shape[0]:
            self.z_pad = tf.zeros([x.shape[0], self.hid_dim])
            self.x_pad = tf.zeros([x.shape[0], self.in_dim])
            self.zf0_tm1 = self.x_pad
            self.zf1_tm1 = self.z_pad
            self.zf2_tm1 = self.z_pad
            self.e1v_tm1 = self.z_pad
            self.e2v_tm1 = self.z_pad
            self.z1 = self.z_pad
            self.z2 = self.z_pad
            self.e1 = self.z_pad
            self.e2 = self.z_pad
            self.ex = self.zf0_tm1
            
        # Initialize current input state
        self.zf0 = x_
        # print("zf0 shape:", self.zf0.shape)
        
        # Compute gates for layer 2
        # print("G2_1 shape:", self.G2_1.shape)
        # print("G2_2 shape:", self.G2_2.shape)
        g2_term1 = tf.matmul(self.zf0, self.G2_1)  # G_2^1 y_0^t
        g2_term2 = tf.matmul(self.zf2_tm1, self.G2_2)  # G_2^2 y_2^{t-1}
        self.g2 = tf.nn.sigmoid(g2_term1 + g2_term2)
        
        # Compute gates for layer 1
        # print("G1_1 shape:", self.G1_1.shape)
        # print("G1_2 shape:", self.G1_2.shape)
        g1_term1 = tf.matmul(self.zf0, self.G1_1)  # G_1^1 y_0^t
        g1_term2 = tf.matmul(self.zf1_tm1, self.G1_2)  # G_1^2 y_1^{t-1}
        self.g1 = tf.nn.sigmoid(g1_term1 + g1_term2)
        
        # Compute layer 2 pre-activation with normalized fast weights
        if self.zeta > 0.0:
            a2 = tf.add(tf.matmul(self.zf1_tm1, self.M2), tf.matmul(self.zf2_tm1, self.V2))
        else:
            a2 = tf.matmul(self.zf2_tm1, self.V2)
        # Add normalized fast weight contribution
        a2 = tf.add(a2, tf.matmul(self.zf2_tm1, self.A2))
        # Apply gate
        self.z2 = self.g2 * a2
        
        if self.standardize:
            self.z2 = standardize(self.z2)
        self.zf2 = self.act_fx(self.z2)
        z1_mu = tf.matmul(self.zf2, self.W2)
        
        # Compute layer 1 pre-activation with normalized fast weights
        if self.zeta > 0.0:
            a1 = tf.add(tf.add(tf.matmul(self.zf0_tm1, self.M1), tf.matmul(self.zf2_tm1, self.U1)), 
                       tf.matmul(self.zf1_tm1, self.V1))
        else:
            a1 = tf.add(tf.matmul(self.zf0_tm1, self.M1), tf.matmul(self.zf1_tm1, self.V1))
        # Add normalized fast weight contribution
        a1 = tf.add(a1, tf.matmul(self.zf1_tm1, self.A1))
        # Apply gate
        self.z1 = self.g1 * a1
        
        if self.standardize:
            self.z1 = standardize(self.z1)
        self.zf1 = self.act_fx(self.z1)
        x_logits = tf.matmul(self.zf1, self.W1)
        x_mu = self.out_fx(x_logits)
        
        # Compute local errors and state perturbations
        self.e1 = tf.subtract(z1_mu, self.zf1) * m
        self.ex = tf.subtract(x_mu, self.zf0) * m
        d2 = tf.matmul(self.e1, self.E2)
        d1 = tf.matmul(self.ex, self.E1)
        
        if self.L1 > 0.0:
            d2 = tf.add(d2, tf.sign(self.z2) * self.L1)
            d1 = tf.add(d1, tf.sign(self.z1) * self.L1)
            
        # Compute state targets
        self.y2 = self.act_fx(tf.subtract(self.z2, d2 * beta))
        self.y1 = self.act_fx(tf.subtract(self.z1, d1 * beta))
        
        # Compute temporal weight corrections
        self.e2v = tf.subtract(self.zf2, self.y2) * m
        self.e1v = tf.subtract(self.zf1, self.y1) * m
        
        # Update fast weights
        if not is_eval:
            self._update_fast_weights()
        
        return x_logits, x_mu

    def clear_var_history(self):
        """
        Clear all state variables including fast weights and gates
        """
        super().clear_var_history()
        self.g1 = None
        self.g2 = None
        # Scale down fast weights with increased decay
        self.A1.assign(self.A1 * self.fast_weight_decay * 0.9)
        self.A2.assign(self.A2 * self.fast_weight_decay * 0.9)

    def collect_params(self):
        """
        Override collect_params to include fast weights and gate parameters
        """
        theta = super().collect_params()
        # Add fast weights
        theta["A1"] = self.A1
        theta["A2"] = self.A2
        # Add gate parameters
        theta["G1_1"] = self.G1_1
        theta["G1_2"] = self.G1_2
        theta["G2_1"] = self.G2_1
        theta["G2_2"] = self.G2_2
        return theta
