from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid

def make_image(X_star, snap, dir_name, u_pred, v_pred, p_pred, u_star, v_star, p_star):
    # 2D用データ
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 200
    x = np.linspace(lb[0], ub[0], nn)
    y = np.linspace(lb[1], ub[1], nn)
    X, Y = np.meshgrid(x,y)
    fig = plt.figure(figsize=(10, 20))
    grid = ImageGrid(fig, 111, nrows_ncols=(3, 3), axes_pad=0.5, label_mode='L',
            cbar_location='right', cbar_mode='each', cbar_pad=0.2) 
    u_diff = u_star - u_pred
    v_diff = v_star - v_pred
    p_diff = p_star - p_pred
    U_pred = griddata(X_star, u_pred.flatten(), (X, Y), method='cubic')
    U_diff = griddata(X_star, u_diff.flatten(), (X, Y), method='cubic')
    U_star = griddata(X_star, u_star.flatten(), (X, Y), method='cubic')
    V_pred = griddata(X_star, v_pred.flatten(), (X, Y), method='cubic')
    V_diff = griddata(X_star, v_diff.flatten(), (X, Y), method='cubic')
    V_star = griddata(X_star, v_star.flatten(), (X, Y), method='cubic')
    P_pred = griddata(X_star, p_pred.flatten(), (X, Y), method='cubic')
    P_diff = griddata(X_star, p_diff.flatten(), (X, Y), method='cubic')
    P_star = griddata(X_star, p_star.flatten(), (X, Y), method='cubic')
    
    u_star_min = str(round(np.amin(u_star), 3))
    u_star_max = str(round(np.amax(u_star), 3))
    v_star_min = str(round(np.amin(v_star), 3))
    v_star_max = str(round(np.amax(v_star), 3))
    p_star_min = str(round(np.amin(p_star), 3))
    p_star_max = str(round(np.amax(p_star), 3))
    
    # 2D表示用
    im = grid[0].pcolor(X, Y, U_star, cmap='coolwarm', norm=Normalize(vmin=u_star_min, vmax=u_star_max))
    grid[0].set_title('U Star'.format(snap))
    cbar = grid.cbar_axes[0].colorbar(im)

    im = grid[1].pcolor(X, Y, U_pred, cmap='coolwarm', norm=Normalize(vmin=u_star_min, vmax=u_star_max))
    grid[1].set_title('U pred'.format(snap))
    cbar = grid.cbar_axes[1].colorbar(im)

    im = grid[2].pcolor(X, Y, U_diff, cmap='coolwarm', norm=Normalize(vmin=u_star_min, vmax=u_star_max))
    grid[2].set_title('U diff'.format(snap))
    cbar = grid.cbar_axes[2].colorbar(im)

    im = grid[3].pcolor(X, Y, V_star, cmap='coolwarm', norm=Normalize(vmin=v_star_min, vmax=v_star_max))
    grid[3].set_title('V Star'.format(snap))
    cbar = grid.cbar_axes[3].colorbar(im)
    
    im = grid[4].pcolor(X, Y, V_pred, cmap='coolwarm', norm=Normalize(vmin=v_star_min, vmax=v_star_max))
    grid[4].set_title('V pred'.format(snap))
    cbar = grid.cbar_axes[4].colorbar(im)

    im = grid[5].pcolor(X, Y, V_diff, cmap='coolwarm', norm=Normalize(vmin=v_star_min, vmax=v_star_max))
    grid[5].set_title('V diff'.format(snap))
    cbar = grid.cbar_axes[5].colorbar(im)
    
    im = grid[6].pcolor(X, Y, P_star, cmap='coolwarm', norm=Normalize(vmin=p_star_min, vmax=p_star_max))
    grid[6].set_title('P Star'.format(snap))
    cbar = grid.cbar_axes[6].colorbar(im)
    cbar.set_clim(p_star_min, p_star_max)

    im = grid[7].pcolor(X, Y, P_pred, cmap='coolwarm')
    grid[7].set_title('P pred'.format(snap))
    cbar = grid.cbar_axes[7].colorbar(im)

    im = grid[8].pcolor(X, Y, P_diff, cmap='coolwarm')
    grid[8].set_title('P diff'.format(snap))
    cbar = grid.cbar_axes[8].colorbar(im)

    plt.show()

    fig_name = dir_name + "/{}_Snap.png".format(snap)

    fig.savefig(fig_name)


def make_loss_translation(dir_name, model):
    fig = plt.figure(figsize=(16, 3))
    plt.xlabel("itr #")
    plt.ylabel("loss")
    plt.yscale("log")
    #plt.ylim(0, 10 ** 7)

    print("final loss values;")
    print(model.loss_log[-1])
    print(model.loss_pred_log[-1])
    print(model.loss_phys_log[-1])

    plt.plot(model.loss_log,      label = "loss")
    plt.plot(model.loss_pred_log, label = "loss_pred", linestyle = ":")
    plt.plot(model.loss_phys_log, label = "loss_phys", linestyle = ":")
    plt.legend(loc = "upper right")
    fig_name = dir_name + "/loss_translation.png"
    fig.savefig(fig_name)


def make_rhoi_translation(dir_name, model):
    fig = plt.figure(figsize=(16, 3))
    plt.xlabel("itr #")
    plt.ylabel("rhoi")
    plt.yscale("log")
    #plt.ylim(0, 10 ** 7)

    print("final loss values;")
    print(model.rhoi_log[-1])

    plt.plot(model.rhoi_log,                    label = "pre")
    plt.hlines(1., 0, len(model.rhoi_log),      label = "ref", linestyle = ":")
    plt.legend(loc = "upper right")
    fig_name = dir_name + "/rhoi_translation.png"
    fig.savefig(fig_name)


def make_nu_translation(dir_name, model):
    fig = plt.figure(figsize=(16, 3))
    plt.xlabel("itr #")
    plt.ylabel("nu")
    plt.yscale("log")
    #plt.ylim(0, 10 ** 7)

    print("final loss values;")
    print(model.nu_log[-1])

    plt.plot(model.nu_log,                    label = "pre")
    plt.hlines(.01, 0, len(model.nu_log),      label = "ref", linestyle = ":")
    plt.legend(loc = "upper right")
    fig_name = dir_name + "/nu_translation.png"
    fig.savefig(fig_name)


def test_images(X_star, snap, dir_name, u_star, v_star, p_star):
    # 2D用データ
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 200
    x = np.linspace(lb[0], ub[0], nn)
    y = np.linspace(lb[1], ub[1], nn)
    X, Y = np.meshgrid(x,y)
    fig = plt.figure(figsize=(10, 20))
    grid = ImageGrid(fig, 111, nrows_ncols=(1, 3), axes_pad=0.5, label_mode='L',
            cbar_location='right', cbar_mode='each', cbar_pad=0.2) 
    U_star = griddata(X_star, u_star.flatten(), (X, Y), method='cubic')
    V_star = griddata(X_star, v_star.flatten(), (X, Y), method='cubic')
    P_star = griddata(X_star, p_star.flatten(), (X, Y), method='cubic')
    
    u_star_min = str(round(np.amin(u_star), 3))
    u_star_max = str(round(np.amax(u_star), 3))
    v_star_min = str(round(np.amin(v_star), 3))
    v_star_max = str(round(np.amax(v_star), 3))
    p_star_min = str(round(np.amin(p_star), 3))
    p_star_max = str(round(np.amax(p_star), 3))
    
    # 2D表示用
    im = grid[0].pcolor(X, Y, U_star, cmap='coolwarm', norm=Normalize(vmin=u_star_min, vmax=u_star_max))
    grid[0].set_title('U Star'.format(snap))
    cbar = grid.cbar_axes[0].colorbar(im)

    im = grid[1].pcolor(X, Y, V_star, cmap='coolwarm', norm=Normalize(vmin=v_star_min, vmax=v_star_max))
    grid[1].set_title('V Star'.format(snap))
    cbar = grid.cbar_axes[1].colorbar(im)
    
    im = grid[2].pcolor(X, Y, P_star, cmap='coolwarm', norm=Normalize(vmin=p_star_min, vmax=p_star_max))
    grid[2].set_title('P Star'.format(snap))
    cbar = grid.cbar_axes[2].colorbar(im)
    cbar.set_clim(p_star_min, p_star_max)

    plt.show()

    fig_name = dir_name + "/{}_Snap.png".format(snap)

    fig.savefig(fig_name)

def test_image(X_star, snap, dir_name, star):
    # 2D用データ
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 200
    x = np.linspace(lb[0], ub[0], nn)
    y = np.linspace(lb[1], ub[1], nn)
    X, Y = np.meshgrid(x,y)
    fig = plt.figure(figsize=(10, 20))
    grid = ImageGrid(fig, 111, nrows_ncols=(1, 1), axes_pad=0.5, label_mode='L',
            cbar_location='right', cbar_mode='each', cbar_pad=0.2) 
    Star = griddata(X_star, star.flatten(), (X, Y), method='cubic')

    star_min = str(round(np.amin(star), 3))
    star_max = str(round(np.amax(star), 3))
    
    # 2D表示用
    im = grid[0].pcolor(X, Y, Star, cmap='coolwarm', norm=Normalize(vmin=star_min, vmax=star_max))
    grid[0].set_title('{} snap'.format(snap))
    cbar = grid.cbar_axes[0].colorbar(im)

    plt.show()

    fig_name = dir_name + "/{}_Snap.png".format(snap)

    fig.savefig(fig_name)