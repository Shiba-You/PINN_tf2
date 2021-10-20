import tensorflow as tf
import numpy as np
import time

class PINN_keras(tf.keras.Model):
    def __init__(self, 
                 x_trn, y_trn, t_trn, u_trn, v_trn, 
                 x_val, y_val, t_val, u_val, v_val, 
                 Rm, Rn, Rl, depth, activ = "tanh",
                 w_init = "glorot_normal", b_init = "zeros", 
                 lr = 1e-3, opt = "Adam", w_prd = 1., w_pde = 1.):
        
        # configurations
        super().__init__()
        self.data_type  = tf.float32
        
        # training set
        self.x_trn = x_trn; self.y_trn = y_trn; self.t_trn = t_trn
        self.u_trn = u_trn; self.v_trn = v_trn
        
        # validation set
        self.x_val = x_val; self.y_val = y_val; self.t_val = t_val
        self.u_val = u_val; self.v_val = v_val
        
        # bounds
        X_trn   = tf.concat([x_trn, y_trn, t_trn], 1)
        self.lb = tf.cast(tf.reduce_min(X_trn, axis = 0), self.data_type)
        self.ub = tf.cast(tf.reduce_max(X_trn, axis = 0), self.data_type)
        
        # network configuration
        self.Rm     = Rm       # input dimension
        self.Rn     = Rn       # output dimension
        self.Rl     = Rl       # internal dimension
        self.depth  = depth    # (# of hidden layers) + output layer
        self.activ  = activ    # activation function
        self.w_init = w_init   # initial weight
        self.b_init = b_init   # initial bias
        
        # optimization setting
        self.lr    = lr        # learning rate
        self.opt   = opt       # name of your optimizer (SGD, RMSprop, Adam, etc.)
        self.w_prd = w_prd     # weight for predictive loss term
        self.w_pde = w_pde     # weight for physical loss term
        
        # parameter setting
        self.rho = tf.constant(1.,  dtype = self.data_type)
        self.nu  = tf.constant(.01, dtype = self.data_type)
#         self.rhoi = tf.Variable(.1, dtype = self.data_type)
#         self.nu   = tf.Variable(.1, dtype = self.data_type)
#         self.rho_log = []
#         self.nu_log  = []

        # track loss
        self.ep_log = []
        self.loss_trn_log = []
        self.loss_val_log = []
        
        # call
        self.dnn = self.dnn_init(Rm, Rn, Rl, depth)
        self.params = self.dnn.trainable_variables
        self.optimizer = self.optimizer(self.lr, self.opt)
        
    def dnn_init(self, Rm, Rn, Rl, depth):
        # network configuration (N: Rm -> Rn (Rm -> Rl -> ... -> Rl -> Rn))
        network = tf.keras.Sequential()
        network.add(tf.keras.layers.InputLayer(Rm))
        network.add(tf.keras.layers.Lambda(lambda x: 2. * (x - self.lb) / (self.ub - self.lb) - 1.))
        for l in range(depth - 1):
            network.add(tf.keras.layers.Dense(Rl, activation = self.activ, use_bias = True,
                                              kernel_initializer = self.w_init, bias_initializer = self.b_init, 
                                              kernel_regularizer = None, bias_regularizer = None, 
                                              activity_regularizer = None, kernel_constraint = None, bias_constraint = None))
        network.add(tf.keras.layers.Dense(Rn))​
        return network
    
    def summary(self):
        return self.dnn.summary()
    
    def optimizer(self, lr, opt):
        if   opt == "SGD":
            optimizer = tf.keras.optimizers.SGD(learning_rate = lr, momentum = 0.0, nesterov = False)
        elif opt == "Adadelta":
            optimizer = tf.keras.optimizers.Adadelta(learning_rate = lr, rho = 0.95)
        elif opt == "Adagrad":
            optimizer = tf.keras.optimizers.Adagrad(learning_rate = lr, initial_accumulator_value = 0.1)
        elif opt == "RMSprop":
            optimizer = tf.keras.optimizers.RMSprop(learning_rate = lr, rho = 0.9, momentum = 0.0, centered = False)
        elif opt == "Adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate = lr, beta_1 = 0.9, beta_2 = 0.999, amsgrad = False)
        elif opt == "Adamax":
            optimizer = tf.keras.optimizers.Adamax(learning_rate = lr, beta_1 = 0.9, beta_2 = 0.999)
        elif opt == "Nadam":
            optimizer = tf.keras.optimizers.Nadam(learning_rate = lr, beta_1 = 0.9, beta_2 = 0.999)
        else:
            raise Exception(">>>>> Exception: optimizer not specified correctly")
            
        return optimizer
    
    def PDE(self, x, y, t):
        rho = self.rho
        nu  = self.nu
        x = tf.convert_to_tensor(x, dtype = self.data_type)
        y = tf.convert_to_tensor(y, dtype = self.data_type)
        t = tf.convert_to_tensor(t, dtype = self.data_type)
        with tf.GradientTape(persistent = True) as tp:
            tp.watch(x)
            tp.watch(y)
            tp.watch(t)
            u_v_p = self.dnn(tf.concat([x, y, t], 1))
            u = u_v_p[:,0:1]
            v = u_v_p[:,1:2]
            p = u_v_p[:,2:3]
            u_x = tp.gradient(u, x); u_y = tp.gradient(u, y)
            v_x = tp.gradient(v, x); v_y = tp.gradient(v, y)
        u_t  = tp.gradient(u, t);   v_t  = tp.gradient(v, t)
        u_xx = tp.gradient(u_x, x); u_yy = tp.gradient(u_y, y)
        v_xx = tp.gradient(v_x, x); v_yy = tp.gradient(v_y, y)
        p_x  = tp.gradient(p, x);   p_y  = tp.gradient(p, y)
        del tp
        gv_c = u_x + v_y                                                  # continuity
        gv_x = u_t + u * u_x + v * u_y + p_x / rho - nu * (u_xx + u_yy)   # momentum
        gv_y = v_t + u * v_x + v * v_y + p_y / rho - nu * (v_xx + v_yy)
        return u, v, p, gv_c, gv_x, gv_y
    
    def loss_prd(self, x, y, t, u, v):
        u_hat, v_hat, p_hat, dummy, dummy, dummy = self.PDE(x, y, t)
        loss_prd =    tf.reduce_mean(tf.square(u - u_hat)) \
                    + tf.reduce_mean(tf.square(v - v_hat))
        loss_prd = self.w_prd * loss_prd
        return loss_prd
        
    def loss_pde(self, x, y, t):
        dummy, dummy, dummy, gv_c_hat, gv_x_hat, gv_y_hat = self.PDE(x, y, t)
        loss_pde =    tf.reduce_mean(tf.square(gv_c_hat)) \
                    + tf.reduce_mean(tf.square(gv_x_hat)) \
                    + tf.reduce_mean(tf.square(gv_y_hat))
        loss_pde = self.w_pde * loss_pde
        return loss_pde
    
    @tf.function
    def loss_glb(self, x, y, t, u, v):
        loss_glb = self.loss_prd(x, y, t, u, v) + self.loss_pde(x, y, t)
        return loss_glb

    def loss_grad(self, x, y, t, u, v):
        with tf.GradientTape(persistent = True) as tp:
            loss = self.loss_glb(x, y, t, u, v)
        grad = tp.gradient(loss, self.params)
        del tp
        return loss, grad
    
    @tf.function
    def grad_desc(self, x, y, t, u, v):
        loss, grad = self.loss_grad(x, y, t, u, v)
        self.optimizer.apply_gradients(zip(grad, self.params))
        return loss
        
    def train(self, epoch = 10 ** 5, batch = 2 ** 6, tol = 1e-5): 
        print(">>>>> training setting;")
        print("         # of epoch     :", epoch)
        print("         batch size     :", batch)
        print("         convergence tol:", tol)
        n_trn = self.x_trn.shape[0]
        n_val = self.x_val.shape[0]
        t0 = time.time()
        for ep in range(epoch):
            ep_loss_trn = 0
            ep_loss_val = 0
            shf_idx_trn = np.random.permutation(n_trn)
            shf_idx_val = np.random.permutation(n_val)
            # training set
            for idx in range(0, n_trn, batch):
                # training set
                x_trn_btch = tf.convert_to_tensor(self.x_trn[shf_idx_trn[idx: idx + batch if idx + batch < n_trn else n_trn]], dtype = self.dtype)
                y_trn_btch = tf.convert_to_tensor(self.y_trn[shf_idx_trn[idx: idx + batch if idx + batch < n_trn else n_trn]], dtype = self.dtype)
                t_trn_btch = tf.convert_to_tensor(self.t_trn[shf_idx_trn[idx: idx + batch if idx + batch < n_trn else n_trn]], dtype = self.dtype)
                u_trn_btch = tf.convert_to_tensor(self.u_trn[shf_idx_trn[idx: idx + batch if idx + batch < n_trn else n_trn]], dtype = self.dtype)
                v_trn_btch = tf.convert_to_tensor(self.v_trn[shf_idx_trn[idx: idx + batch if idx + batch < n_trn else n_trn]], dtype = self.dtype)
                # compute loss and perform gradient descent
                loss_trn = self.grad_desc(x_trn_btch, y_trn_btch, t_trn_btch, u_trn_btch, v_trn_btch)
                ep_loss_trn += loss_trn / int(n_trn / batch)
            # validation set
            for idx in range(0, n_val, batch):
                # validation set
                x_val_btch = tf.convert_to_tensor(self.x_val[shf_idx_val[idx: idx + batch if idx + batch < n_val else n_val]], dtype = self.dtype)
                y_val_btch = tf.convert_to_tensor(self.y_val[shf_idx_val[idx: idx + batch if idx + batch < n_val else n_val]], dtype = self.dtype)
                t_val_btch = tf.convert_to_tensor(self.t_val[shf_idx_val[idx: idx + batch if idx + batch < n_val else n_val]], dtype = self.dtype)
                u_val_btch = tf.convert_to_tensor(self.u_val[shf_idx_val[idx: idx + batch if idx + batch < n_val else n_val]], dtype = self.dtype)
                v_val_btch = tf.convert_to_tensor(self.v_val[shf_idx_val[idx: idx + batch if idx + batch < n_val else n_val]], dtype = self.dtype)
                # only compute loss, w/o gradient descent
                loss_val = self.loss_glb (x_val_btch, y_val_btch, t_val_btch, u_val_btch, v_val_btch)
                ep_loss_val += loss_val / int(n_val / batch)
            if ep % f_mntr == 0:
                elps = time.time() - t0
                self.ep_log.append(ep)
                self.loss_trn_log.append(ep_loss_trn)
                self.loss_val_log.append(ep_loss_val)
                print("ep: %d, loss_trn: %.3e, loss_val: %.3e, elps: %.3f" % (ep, ep_loss_trn, ep_loss_val, elps))
                t0 = time.time()
            if ep_loss_trn < tol:
                print(">>>>> program terminating with the loss converging to its tolerance.")
                print("\n************************************************************")
                print("*****************     MAIN PROGRAM END     *****************")
                print("************************************************************")
                print(">>>>> end time:", datetime.datetime.now())
                break
        print("\n************************************************************")
        print("*****************     MAIN PROGRAM END     *****************")
        print("************************************************************")
        print(">>>>> end time:", datetime.datetime.now())
                
    def predict(self, x, y, t):
        u_hat, v_hat, p_hat, gv_c_hat, gv_x_hat, gv_y_hat = self.PDE(x, y, t)
        return u_hat, v_hat, p_hat, gv_c_hat, gv_x_hat, gv_y_hat