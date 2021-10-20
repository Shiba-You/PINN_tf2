import pandas as pd
import numpy as np
from make_modes import make_modes

# def make_data(**kwargs):
#     '''
#     kwargs = {
#         path    : データまでのpath  
#         N_train : N*T*N_train が最終的なトレーニングデータ数となる
#         subject : どの物理量を対象にPOD分解を実施するのか
#         n_modes : 第何モードまでを対象とするのか
#         mode_th : 各モード諸条件の上位何%を教師データ候補群とするのか
#     }

#     XX      = (N, T)      ,  YY      = (N, T)      ,  TT      = (N, T)      ,  UU      = (N, T)      ,  VV      = (N, T)      ,  PP      = (N, T)      
#     x       = (N*T, 1)    ,  y       = (N*T, 1)    ,  t       = (N*T, 1)    ,  u       = (N*T, 1)    ,  v       = (N*T, 1)    ,  p       = (N*T, 1)    
#     x_train = (N_train, 1),  y_train = (N_train, 1),  t_train = (N_train, 1),  u_train = (N_train, 1),  v_train = (N_train, 1),  p_train = (N_train, 1)
#     '''
#     U_star, P_star, Vor_star             = read_data(kwargs["path"])
#     X_star, t_star, N, T, dx, dy, nx, ny = init_data()
        

#     XX = np.tile(X_star[:,0:1], (1,T)) # N x T
#     YY = np.tile(X_star[:,1:2], (1,T)) # N x T
#     TT = np.tile(t_star, (1,N)).T # N x T
#     UU = U_star[:,0].reshape(T, N).T            # N x T X軸方向速度
#     VV = - U_star[:,1].reshape(T, N).T        # N x T Y軸方向速度
#     PP = P_star.reshape(T, N).T
#     WW = Vor_star[:,0].reshape(T, N).T            # N x T X軸方向速度

#     x = XX.flatten()[:,None] # NT x 1
#     y = YY.flatten()[:,None] # NT x 1
#     t = TT.flatten()[:,None] # NT x 1
#     u = UU.flatten()[:,None] # NT x 1
#     v = VV.flatten()[:,None] # NT x 1
#     p = PP.flatten()[:,None] # NT x 1

#     if kwargs["n_modes"] != 0:
#         subject = eval(kwargs["subject"])
#         cand_idx = make_modes(subject, kwargs["n_modes"], N, T, kwargs["mode_th"], dx, dy, nx, ny)
#         idx = np.random.choice(cand_idx, int(kwargs["N_train"]*N*T), replace=False)
#         idx = idx.astype("int64")
#     else:
#         idx = np.random.choice(N*T, int(kwargs["N_train"]*N*T), replace=False)
#     x_train = x[idx,:]
#     y_train = y[idx,:]
#     t_train = t[idx,:]
#     u_train = u[idx,:]
#     v_train = v[idx,:]

#     log(x_train, y_train, t_train, u_train, v_train)

#     return X_star, x_train, y_train, t_train, u_train, v_train, TT, UU, VV, PP


def read_data(path):
    U_star = pd.read_csv(path + 'U_arrange_data.csv', header = None, sep = '\s+').values
    P_star = pd.read_csv(path + 'p_arrange_data.csv', header = None, sep = '\s+').values
    Vor_star = pd.read_csv(path + 'vorticity_arrange_data.csv', header = None, sep = '\s+').values
    return U_star, P_star, Vor_star

def init_data():
    t0, t1, dt =  0, 20, .1
    t_star = np.arange(t0, t1, dt)
    t_star = t_star.reshape(-1, 1)

    x0, x1, nx =  1,  8, 99
    x_star, dx = np.linspace(x0, x1, nx, retstep=True)   # true domain: [1.075, 7.950]

    y0, y1, ny = -2,  2, 50
    y_star, dy = np.linspace(y0, y1, ny, retstep=True)   # true domain: [-1.942, 1.943]

    x_star, y_star = np.meshgrid(x_star, y_star)
    x_star, y_star = x_star.reshape(-1, 1), y_star.reshape(-1, 1)
    X_star = np.c_[x_star, y_star]

    N = X_star.shape[0]
    T = t_star.shape[0]

    return X_star, t_star, N, T, dx, dy, nx, ny

def log(x_train, y_train, t_train, u_train, v_train, x_valid, y_valid, t_valid, u_valid, v_valid):
    print("x_train : ", x_train.shape, "    x_valid : ", x_valid.shape)
    print("y_train : ", y_train.shape, "    y_valid : ", y_valid.shape)
    print("t_train : ", t_train.shape, "    t_valid : ", t_valid.shape)
    print("u_train : ", u_train.shape, "    u_valid : ", u_valid.shape)
    print("v_train : ", v_train.shape, "    v_valid : ", v_valid.shape)



def make_data(**kwargs):
    '''
    kwargs = {
        path    : データまでのpath  
        N_train : N*T*N_train が最終的なトレーニングデータ数となる
        subject : どの物理量を対象にPOD分解を実施するのか
        n_modes : 第何モードまでを対象とするのか
        mode_th : 各モード諸条件の上位何%を教師データ候補群とするのか
        ns_lv   : ノイズの大きさ
    }

    XX      = (N, T)      ,  YY      = (N, T)      ,  TT      = (N, T)      ,  UU      = (N, T)      ,  VV      = (N, T)      ,  PP      = (N, T)      
    x       = (N*T, 1)    ,  y       = (N*T, 1)    ,  t       = (N*T, 1)    ,  u       = (N*T, 1)    ,  v       = (N*T, 1)    ,  p       = (N*T, 1)    
    x_train = (N_train, 1),  y_train = (N_train, 1),  t_train = (N_train, 1),  u_train = (N_train, 1),  v_train = (N_train, 1),  p_train = (N_train, 1)
    '''
    U_star, P_star, Vor_star             = read_data(kwargs["path"])
    X_star, t_star, N, T, dx, dy, nx, ny = init_data()
        

    XX = np.tile(X_star[:,0:1], (1,T)) # N x T
    YY = np.tile(X_star[:,1:2], (1,T)) # N x T
    TT = np.tile(t_star, (1,N)).T # N x T
    UU = U_star[:,0].reshape(T, N).T            # N x T X軸方向速度
    VV = - U_star[:,1].reshape(T, N).T        # N x T Y軸方向速度
    PP = P_star.reshape(T, N).T
    WW = Vor_star[:,0].reshape(T, N).T            # N x T X軸方向速度

    UU = UU + kwargs["ns_lv"] * np.std(UU) * np.random.randn(UU.shape[0], UU.shape[1])
    VV = VV + kwargs["ns_lv"] * np.std(VV) * np.random.randn(VV.shape[0], VV.shape[1])
    PP = PP + kwargs["ns_lv"] * np.std(PP) * np.random.randn(PP.shape[0], PP.shape[1])

    x = XX.flatten()[:,None] # NT x 1
    y = YY.flatten()[:,None] # NT x 1
    t = TT.flatten()[:,None] # NT x 1
    u = UU.flatten()[:,None] # NT x 1
    v = VV.flatten()[:,None] # NT x 1
    p = PP.flatten()[:,None] # NT x 1

    N_train = int(kwargs["N_train"]*N*T)
    N_valid = int(kwargs["N_valid"]*N*T)
    N_all = N_train + N_valid

    idx_all = np.random.choice(N * T, N_all, replace = False)
    idx_trn = idx_all[0 : N_train]
    idx_val = idx_all[N_train : N_all]

    if kwargs["n_modes"] != 0:
        subject = eval(kwargs["subject"])
        cand_idx = make_modes(subject, kwargs["n_modes"], N, T, kwargs["mode_th"], dx, dy, nx, ny)

        idx_all = np.random.choice(cand_idx, N_all, replace = False).astype("int64")
        # idx = np.random.choice(cand_idx, int(kwargs["N_train"]*N*T), replace=False)
        # idx = idx.astype("int64")

    else:
        idx_all = np.random.choice(N*T, N_all, replace = False).astype("int64")
        # idx = np.random.choice(N*T, int(kwargs["N_train"]*N*T), replace=False)
    
    idx_train = idx_all[0 : N_train]
    idx_valid = idx_all[N_train : N_all]

    x_train = x[idx_train,:]
    y_train = y[idx_train,:]
    t_train = t[idx_train,:]
    u_train = u[idx_train,:]
    v_train = v[idx_train,:]

    x_valid = x[idx_valid,:]
    y_valid = y[idx_valid,:]
    t_valid = t[idx_valid,:]
    u_valid = u[idx_valid,:]
    v_valid = v[idx_valid,:]


    log(x_train, y_train, t_train, u_train, v_train, x_valid, y_valid, t_valid, u_valid, v_valid)

    return X_star, x_train, y_train, t_train, u_train, v_train, x_valid, y_valid, t_valid, u_valid, v_valid, TT, UU, VV, PP