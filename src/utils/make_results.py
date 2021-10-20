import os
import datetime
from make_images import make_image
from make_images import make_loss_translation, make_rhoi_translation, make_nu_translation
import numpy as np


def make_results(pro, subject, model, X_star, TT, snap, UU, VV, PP, n_modes, mode_th, N_train, Itration, elps, rs, ns_lv, alpha):
    dir_name = make_dir(pro, subject, N_train, ns_lv, alpha)
    u_pred, v_pred, p_pred, u_star, v_star, p_star, error_u, error_v, error_p, error_lambda_1, error_lambda_2, N, T = model_pred(model, X_star, TT, snap, UU, VV, PP)
    train_data = int(N * T * N_train)
    make_info(dir_name=dir_name, \
            project=pro, \
            subject=subject, \
            Itration=Itration, \
            random_seed=rs,\
            train_data=train_data, \
            ns_lv=ns_lv, \
            alpha=alpha, \
            n_modes=n_modes, \
            mode_th=mode_th, \
            elps=elps, \
            error_u=error_u,\
            error_v=error_v,\
            error_p=error_p,\
            error_lambda_1=error_lambda_1,\
            error_lambda_2=error_lambda_2
    )
    make_image(X_star, snap, dir_name, u_pred, v_pred, p_pred, u_star, v_star, p_star)
    make_loss_translation(dir_name, model)
    make_rhoi_translation(dir_name, model)
    make_nu_translation(dir_name, model)



def make_dir(pro, subject, train_rate, ns_lv, alpha):
    dir_pro = "../../../output/{}".format(pro)
    dir_name = dir_pro + "/{}_{}_{}_{}_{}".format(subject, train_rate, ns_lv, alpha, str(datetime.date.today()))
    if not os.path.exists(pro):
        os.mkdir(pro)
    if not os.path.exists(dir_name):#ディレクトリがなかったら
        os.mkdir(dir_name)#作成したいフォルダ名を作成
    return dir_name

def model_pred(model, X_star, TT, snap, UU, VV, PP):
    x_star = X_star[:,0:1]
    y_star = X_star[:,1:2]
    t_star = TT[:,snap]

    u_star = UU[:,snap]
    v_star = VV[:,snap]
    p_star = PP[:,snap]
    
    u_pred, v_pred, p_pred = model.predict(x_star, y_star, t_star)
    lambda_1_value = model.sess.run(model.lambda_1)
    lambda_2_value = model.sess.run(model.lambda_2)

    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    error_v = np.linalg.norm(v_star-v_pred,2)/np.linalg.norm(v_star,2)
    error_p = np.linalg.norm(p_star-p_pred,2)/np.linalg.norm(p_star,2)
    error_lambda_1 = np.abs(lambda_1_value - 1.0)*100
    error_lambda_2 = np.abs(lambda_2_value - 0.01)/0.01 * 100

    N = X_star.shape[0]
    T = t_star.shape[0]

    return u_pred, v_pred, p_pred, u_star, v_star, p_star, error_u, error_v, error_p, error_lambda_1, error_lambda_2, N, T

def make_info(**kwargs):

    f = open(kwargs["dir_name"]+"/info.txt", "a", encoding="UTF-8")
    kwargs.pop("dir_name")
    f.write("\n")
    f.write("********************************\n")
    f.write("********************************\n")
    f.write("Date".ljust(15) + ": {} \n".format(str(datetime.datetime.now())))
    for key, val in kwargs.items():
        if key == "error_u" or key == "error_v" or key == "error_p" or key == "elps":
            f.write(key.ljust(15) + ': %e \n' % (val))
        elif key == "error_lambda_1" or key == "error_lambda_2":
            f.write(key.ljust(15) + ': %.5f%% \n' % (val))
        else:
            f.write(key.ljust(15) + ': {} \n'.format(val))
            if key == "mode_th":
                f.write("============Results==============\n")   
    f.write("********************************\n")
    f.close()
