import numpy as np
import matplotlib.pyplot as plt

def make_modes(subject, n_modes, N, T, mode_th, dx, dy, nx, ny):
    s_modes = mode_decomp(subject, n_modes)
    tmp_idx = scrape_modes(s_modes, mode_th, dx, dy, nx, ny)
    idx = choice_coordinate(tmp_idx, N, T)
    return idx

def mode_decomp(subject, n_modes):
    '''
    s_modes_star = (N, T),   weights_star = (T, ) (対角成分) ,   t_modes_star = (T, T)
    s_modes = (N, s_modes),  weights = (s_modes, ) (対角成分),   t_modes = (s_modes, T)
    '''
    s_modes_star, weights_star, t_modes_star = np.linalg.svd(subject, full_matrices = False, compute_uv = True)

    s_modes = s_modes_star[:,:n_modes]
    # weights = weights_star[:n_modes]
    # t_modes = t_modes_star[:n_modes,:]
    return s_modes

def scrape_modes(s_modes, mode_th, dx, dy, nx, ny):
    '''
    grad = (2, N, mode数) (2 = (行方向勾配, 列方向勾配)の順番って書いてるけど，多分（列方向，行方向））
    参考：https://runebook.dev/ja/docs/numpy/reference/generated/numpy.gradient
    '''
    sorted_idx = np.array([])

    for i in range(s_modes.shape[1]):
        mode = np.ravel(abs(s_modes[:,i:i+1])).reshape(ny, nx)
        y_grad, x_grad = np.array(np.gradient(mode, dy, dx))
        grad = np.ravel(np.sqrt(y_grad**2+x_grad**2))
        unsorted_idx_max = np.argpartition(-grad, int(grad.shape[0]*mode_th))[:int(grad.shape[0]*mode_th)]
        unsorted_idx_min = np.argpartition(grad, int(grad.shape[0]*mode_th))[:int(grad.shape[0]*mode_th)]
        unsorted_idx = np.r_[unsorted_idx_max, unsorted_idx_min]
        max_y = grad[unsorted_idx_max]
        min_y = grad[unsorted_idx_min]
        y = np.r_[max_y, min_y]
        idx = np.argsort(-y)
        sorted_idx = np.append(sorted_idx, unsorted_idx[idx])
    sorted_idx = np.unique(sorted_idx)
    scrape_x = np.array([])
    scrape_y = np.array([])
    for val in sorted_idx:
        scrape_x = np.append(scrape_x, val % nx)
        scrape_y = np.append(scrape_y, val // nx)

    fig, ax1 = plt.subplots()
    ax1.scatter(scrape_x, scrape_y)
    ax1.set_ylim([50, 0])
    return sorted_idx


def choice_coordinate(f_idx, N, T):
    cand_idx = np.array([])
    for i in range(T):
        tmp = f_idx+N*i
        cand_idx = np.append(cand_idx, tmp)
    return cand_idx

