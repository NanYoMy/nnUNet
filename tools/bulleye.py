import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker


def bulls_eye(ax, data, cmap=None, norm=None, raidal_subdivisions=(2, 8, 8, 11),
              centered=(False, False, False, False), add_nomenclatures=True, cell_resolution=128,
              pfi_where_to_save=None, colors_bound='-k'):
    """
    Clockwise, from smaller radius to bigger radius.
    :param ax:
    :param data:
    :param cmap:
    :param norm:
    :param raidal_subdivisions:
    :param centered:
    :param add_nomenclatures:
    :param cell_resolution:
    :param pfi_where_to_save:
    :return:
    """
    line_width = 0.5
    data = np.array(data).ravel()

    if cmap is None:
        cmap = plt.cm.viridis

    if norm is None:
        norm = mpl.colors.Normalize(vmin=data.min(), vmax=data.max())

    theta = np.linspace(0, 2*np.pi, 768)
    r = np.linspace(0, 1, len(raidal_subdivisions)+1)

    nomenclatures = []
    if isinstance(add_nomenclatures, bool):
        if add_nomenclatures:
            nomenclatures = range(1, sum(raidal_subdivisions)+1)
    elif isinstance(add_nomenclatures, list) or isinstance(add_nomenclatures, tuple):
        assert len(add_nomenclatures) == sum(raidal_subdivisions)
        nomenclatures = add_nomenclatures[:]
        add_nomenclatures = True


    # Create the circular bounds
    line_width_circular = line_width
    for i in range(r.shape[0]):
        if i == range(r.shape[0])[-1]:
            line_width_circular = int(line_width / 2.)
        ax.plot(theta, np.repeat(r[i], theta.shape), colors_bound, lw=line_width_circular,)
    # plt.show()
    # iterate over cells divided by radial subdivision
    for rs_id, rs in enumerate(raidal_subdivisions):
        for i in range(rs):
            cell_id = sum(raidal_subdivisions[:rs_id]) + i
            # theta_i = - i * 2 * np.pi / rs + np.pi / 2
            theta_i = - i * 2 * np.pi /rs
            if not centered[rs_id]:
                theta_i += (2 * np.pi / rs) / 2
            theta_i_plus_one = theta_i - 2 * np.pi / rs  # clockwise
            # Create colour fillings for each cell:
            theta_interval = np.linspace(theta_i, theta_i_plus_one, cell_resolution)
            r_interval = np.array([r[rs_id], r[rs_id+1]])
            angle  = np.repeat(theta_interval[:, np.newaxis], 2, axis=1)
            radius = np.repeat(r_interval[:, np.newaxis], cell_resolution, axis=1).T
            z = np.ones((cell_resolution, 2)) * data[cell_id]
            ax.pcolormesh(angle, radius, z, cmap=cmap, norm=norm)

            # Create radial bounds
            if rs  > 1:
                ax.plot([theta_i, theta_i], [r[rs_id], r[rs_id+1]], colors_bound, lw=line_width,)
            # Add centered nomenclatures if needed
            if add_nomenclatures:
                if rs == 1 and rs_id ==0:
                    cell_center = (0, 0)
                else:
                    cell_center = ((theta_i + theta_i_plus_one) / 2., r[rs_id] + .5 * r[1] )

                if isinstance(nomenclatures[0], (int,  float, complex)):
                    ax.annotate(r"${:.3g}$".format(nomenclatures[cell_id]), xy=cell_center,
                                xytext=(cell_center[0], cell_center[1]),
                                horizontalalignment='center', verticalalignment='center', size=8)
                else:
                    ax.annotate(nomenclatures[cell_id], xy=cell_center,
                                xytext=(cell_center[0], cell_center[1]),
                                horizontalalignment='center', verticalalignment='center', size=12)

    ax.grid(False)
    ax.set_ylim([0, 1])
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    if pfi_where_to_save is not None:
        plt.savefig(pfi_where_to_save, format='pdf', dpi=200)


def multi_bull_eyes(multi_data, cbar=None, cmaps=None, normalisations=None,
                    global_title=None, canvas_title='title', titles=None, units=None, raidal_subdivisions=(2, 8, 8, 11),
                    centered=(True, True, True, True), add_nomenclatures=(True, True, True, True),
                    pfi_where_to_save=None, show=True):
    plt.clf()
    n_fig = len(multi_data)
    if cbar is None:
        cbar = [True] * n_fig
    if cmaps is None:
        cmaps = [mpl.cm.viridis] * n_fig
    if normalisations is None:
        normalisations = [mpl.colors.Normalize(vmin=np.min(multi_data[i]), vmax=np.max(multi_data[i]))
                          for i in range(n_fig)]
    if titles is None:
        titles = ['Title {}'.format(i) for i in range(n_fig)]

    h_space = 0.15 / n_fig
    h_dim_fig = .8
    w_dim_fig = .8 / n_fig

    def fmt(x, pos):
        # a, b = '{:.2e}'.format(x).split('e')
        # b = int(b)
        # return r'${} \times 10^{{{}}}$'.format(a, b)
        return r"${:.4g}$".format(x)

    # Make a figure and axes with dimensions as desired.
    fig = plt.figure(figsize=(3 * n_fig, 4))
    fig.canvas.set_window_title(canvas_title)
    if global_title is not None:
        plt.suptitle(global_title)

    for n in range(n_fig):
        origin_fig = (h_space * (n + 1) + w_dim_fig * n, 0.15)
        ax = fig.add_axes([origin_fig[0], origin_fig[1], w_dim_fig, h_dim_fig], polar=True)
        bulls_eye(ax, multi_data[n], cmap=cmaps[n], norm=normalisations[n], raidal_subdivisions=raidal_subdivisions,
                  centered=centered, add_nomenclatures=add_nomenclatures[n])
        ax.set_title(titles[n], size=10)

        if cbar[n]:
            origin_cbar = (h_space * (n + 1) + w_dim_fig * n, .15)
            axl = fig.add_axes([origin_cbar[0], origin_cbar[1], w_dim_fig, .05])
            cb1 = mpl.colorbar.ColorbarBase(axl, cmap=cmaps[n], norm=normalisations[n], orientation='horizontal',
                                            format=ticker.FuncFormatter(fmt))
            cb1.ax.tick_params(labelsize=8)
            if units is not None:
                cb1.set_label(units[n])

    if pfi_where_to_save is not None:
        plt.savefig(pfi_where_to_save, format='pdf', dpi=330)
    if show:
        plt.show()


def plot_transmularity_bulleye(data,path,raidal_subdivisions=[1, 33, 33, 33],color_bar=False):
    fig = plt.figure(figsize=(5, 7))
    # fig.canvas.set_window_title('Transmularity')

    # First and only:
    # print(cmap)
    # cmap = mpl.cm.Greys
    # cmap = mpl.cm.tab10
    # cmap = mpl.cm.cool

    # cmap = mpl.cm.bwr
    # norm = mpl.colors.Normalize(vmin=0, vmax=1)

    # cmap = mpl.colors.ListedColormap(["darkcyan", "coral", "orangered", "red"])
    cmap = mpl.colors.ListedColormap(["mistyrose", "coral", "orangered", "red"])
    norm = mpl.colors.BoundaryNorm(np.array([0.0,0.25,0.5,0.75,1.0]), cmap.N)
    #

    ax = fig.add_axes([0.1, 0.2, 0.8, 0.7], polar=True)

    fig.patch.set_facecolor('xkcd:cloudy blue')

    bulls_eye(ax, data, cmap=cmap, norm=norm, raidal_subdivisions=raidal_subdivisions,add_nomenclatures=False,colors_bound='black')
    # ax.set_title('Bulls Eye')

    if color_bar:
        axl = fig.add_axes([0.1, 0.15, 0.8, 0.05])
        cb1 = mpl.colorbar.ColorbarBase(axl, cmap=cmap, norm=norm, orientation='horizontal')
        cb1.set_label('Transmularity')

    # plt.show()
    fig.savefig(path, dpi=400, bbox_inches='tight', pad_inches=0.03)

    return path

if __name__ == '__main__':

    # Very dummy data:
    data = np.array(range(99)) + 1

    # TEST bull-eye three-fold
    if True:

        fig, ax = plt.subplots(figsize=(12, 8), nrows=1, ncols=3,
                               subplot_kw=dict(projection='polar'))
        fig.canvas.set_window_title('Left Ventricle Bulls Eyes')

        # First one:
        cmap = mpl.cm.viridis
        cmap = mpl.colormaps.hsv
        norm = mpl.colors.Normalize(vmin=1, vmax=100)

        bulls_eye(ax[0], data, cmap=cmap, norm=norm,raidal_subdivisions=[0,33,33,33])
        ax[0].set_title('Bulls Eye ')

        axl = fig.add_axes([0.14, 0.15, 0.2, 0.05])
        cb1 = mpl.colorbar.ColorbarBase(axl, cmap=cmap, norm=norm, orientation='horizontal')
        cb1.set_label('Some Units')

        # Second one
        cmap2 = mpl.cm.cool
        norm2 = mpl.colors.Normalize(vmin=1, vmax=29)

        bulls_eye(ax[1], data, cmap=cmap2, norm=norm2)
        ax[1].set_title('Bulls Eye ')

        axl2 = fig.add_axes([0.41, 0.15, 0.2, 0.05])
        cb2 = mpl.colorbar.ColorbarBase(axl2, cmap=cmap2, norm=norm2, orientation='horizontal')
        cb2.set_label('Some other units')

        # Third one
        cmap3 = mpl.cm.winter
        norm3 = mpl.colors.Normalize(vmin=1, vmax=29)

        bulls_eye(ax[2], data, cmap=cmap3, norm=norm3)
        ax[2].set_title('Bulls Eye third')

        axl3 = fig.add_axes([0.69, 0.15, 0.2, 0.05])
        cb3 = mpl.colorbar.ColorbarBase(axl3, cmap=cmap3, norm=norm3, orientation='horizontal')
        cb3.set_label('Some more units')

        plt.show()

    if True:
        fig = plt.figure(figsize=(5, 7))
        fig.canvas.set_window_title('Bulls Eyes - segmentation assessment')

        # First and only:
        cmap = mpl.cm.viridis
        norm = mpl.colors.Normalize(vmin=1, vmax=29)

        ax = fig.add_axes([0.1, 0.2, 0.8, 0.7], polar=True)
        bulls_eye(ax, data, cmap=cmap, norm=norm,raidal_subdivisions=[0,33,33,33])
        ax.set_title('Bulls Eye')

        axl = fig.add_axes([0.1, 0.15, 0.8, 0.05])
        cb1 = mpl.colorbar.ColorbarBase(axl, cmap=cmap, norm=norm, orientation='horizontal')
        cb1.set_label('Some Units')

        plt.show()

    if True:

        multi_data = [range(1,17), list( 0.000000001 * np.array(range(1,17))), list( 0.001 * np.array(range(1,17)))]
        print(multi_data)
        multi_bull_eyes(multi_data, raidal_subdivisions=(3,3,4,6),
                  centered=(True, True, True, True), add_nomenclatures=[True]*3)

        plt.show(block=True)