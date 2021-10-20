# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from os import rmdir
import sys
from threading import active_count
from typing import Optional
from tensorflow.python.eager.function import BACKWARD_FUNCTION_ATTRIBUTE_NAME

from tensorflow.python.ops.gen_array_ops import depth_to_space
sys.path.append("../../utils")
import warnings
warnings.simplefilter('ignore')
import importlib

from pinns import PINN
from make_data import make_data
from send_line import send_line
from make_results import make_results
from comfirm_gpu import comfirm_gpu


import numpy as np
import tensorflow as tf
import time


# %%

# ==============================================================
# #######################  Select Data #########################  
# ==============================================================
'''
pro     : 対象となる例題            # { asymmetric_squares: 非対称,  circle: 円柱,  square: 四角柱 }
path    : 例題データまでのディレクトリ
subject : 出力ファイルの名前
'''
pro = "asymmetric_squares"
path = "../../../data/{}/".format(pro)
subject = "basic"

# ==============================================================
# #######################  Model Param #########################
# ==============================================================
'''
lr      : 学習率                # 1e-3,
c_tol   : 許容誤差              # 1e-3,
n_epch  : エポック数            # int(5e3)
f_mntr  : ログ出力頻度          # int(1e1)
N_train : 教師データの割合      # .007
N_valid : 検証データの割合      # .003
Rm      : 入力層次元            # 3
Rn      : 出力層次元            # 2
Rl      : 中間層次元            # 20
depth   : 中間層数              # 7
activ   : 活性化関数            # tanh      { tanh, elu, gelu, relu, silu, swish...} https://www.tensorflow.org/api_docs/python/tf/keras/activations
w_init  : 重み初期化            # glorot_normal     https://www.tensorflow.org/api_docs/python/tf/keras/initializers
b_init  : バイアス初期化        # zeros    
opt     : 最適化アルゴリズム    # Adam      { SGD, Adadelta, Adagrad, RMSprop, Adam, Adamax, Nadam }       
gpu_flg : GPU出力               # 0                 https://www.tensorflow.org/guide/gpu
seed    : ランダムシード        # 1234
- layers  : 全体の層
'''
lr          = 1e-3              ;c_tol      = 1e-3
n_epch      = int(5e3)          ;f_mntr     = int(1e1)
N_train     = .007              ;N_valid    = .003
Rm          = 3                 ;Rn         = 2             ;Rl     = 20
depth       = 7                 ;activ      = "tanh"
w_init      = "glorot_normal"   ;b_init     = "zeros"       ;opt    = "Adam"
gpu_flg     = 0                 ;seed       = 1234

np.random.seed(seed)   ;tf.random.set_seed(seed)
# layers  = [3] + 15 * [20] + [3]

# ==============================================================
# #######################  Option Param ########################
# ==============================================================
'''
n_batch : バッチサイズ(2^n)     # 7,        { 0: フルバッチ学習 }
ns_lv   : ノイズ率              # 0         { 0: ノイズなし学習, ~ , 1: ノイズ率100% }
n_modes : モード分解数          # 0         { 0: モード分解なし, ~ , N: Nモード分解 }

'''
n_batch  = 0
ns_lv = 0
n_modes = 0
mode_th = 0


# %%
X_star, x_train, y_train, t_train, u_train, v_train, x_valid, y_valid, t_valid, u_valid, v_valid, TT, UU, VV, PP = \
    make_data(path=path, ns_lv=ns_lv, N_train=N_train, N_valid=N_valid, n_modes=n_modes, subject=subject, mode_th=mode_th)

# %%

comfirm_gpu(gpu_flg)


# %%

model = PINN(x_train, y_train, t_train, u_train, v_train,
               x_valid, y_valid, t_valid, u_valid, v_valid, 
               Rm, Rn, Rl, depth, activ, w_init, b_init,
               lr, opt, w_prd = 1., w_pde = 1.)

# %%
with tf.device("GPU:0"):
    model.train(epoch = n_epch, batch=n_batch, tol = c_tol, f_mntr = f_mntr)

        
# %%
