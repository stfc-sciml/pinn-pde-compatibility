import matplotlib.pyplot as plt
import torch
from matplotlib import gridspec

from PDE import pde_obj_dict
from network import MembraneNet

# set plot env
plt.style.use(['seaborn-paper'])
plt.rcParams.update({
    "xtick.major.pad": 2,
    "ytick.major.pad": 1,
    "font.family": "Times",
})

# set TeX
plt.rcParams["text.usetex"] = True
try:
    plt.text(0, 0, '$x$')
    plt.close()
except:
    # if latex is not installed, disable
    plt.rcParams["text.usetex"] = False

if __name__ == "__main__":
    # compute predictions
    seed = 0
    n = 400
    pred_dict = {}
    for pde in pde_obj_dict.keys():
        for m_type in ['vanilla', 'OLHP']:
            model = MembraneNet([8, 16, 32], OLHP=m_type == 'OLHP', seed=seed)
            model.load_state_dict(torch.load(
                f'results/{m_type}_{pde}_seed{seed}/'
                f'model_weights__epoch=1000.pt', map_location='cpu'))
            (u_Omega_pred, u_Gamma_pred, PDE_Omega_pred,
             L_Omega, L_Gamma, L_PDE) = model.predict_PDE(pde_obj_dict[pde],
                                                          n_side=n,
                                                          batch_size=10000)
            pred_dict[f'{pde}_{m_type}'] = u_Omega_pred.reshape(n, n)

    for pde in pde_obj_dict.keys():
        fig = plt.figure(dpi=200, figsize=(4, 4))
        outer = gridspec.GridSpec(2, 1, height_ratios=[1.2, 2], hspace=0)
        ###########
        # history #
        ###########
        inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[1])
        ax = plt.Subplot(fig, inner[0])

        hist_vanilla = torch.load(
            f'results/vanilla_{pde}_seed{seed}/training_history__epoch=1000.pt')
        hist_OLHP = torch.load(
            f'results/OLHP_{pde}_seed{seed}/training_history__epoch=1000.pt')
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        ep = range(0, len(hist_vanilla['L_Omega']) * 5, 5)
        ax.plot(ep, hist_vanilla['L_Omega'], label='Vanilla, $L_\mathrm{data}$',
                c=colors[0], ls='--')
        ax.plot(ep, hist_vanilla['L_Gamma'], label='Vanilla, $L_\mathrm{BC}$',
                c=colors[1], ls='--')
        ax.plot(ep, hist_vanilla['L_PDE'], label='Vanilla, $L_\mathrm{PDE}$',
                c=colors[2], ls='--')
        ax.plot(ep, hist_OLHP['L_Omega'], label='OLHP, $L_\mathrm{data}$',
                c=colors[0])
        ax.plot(ep, hist_OLHP['L_Gamma'], label='OLHP, $L_\mathrm{BC}$',
                c=colors[1])
        ax.set_yscale('log')
        ax.tick_params(axis='both', which='major', labelsize=12)
        if pde == 'smooth':
            ax.legend(fontsize=11, labelspacing=.3)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        fig.add_subplot(ax)

        inner = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0],
                                                 wspace=0.1)

        #########
        # truth #
        #########
        n = 400
        (xy_Omega, u_Omega, K_Omega, Kx_Omega, Ky_Omega, f_Omega,
         xy_Gamma, u_Gamma, K_Gamma, Kx_Gamma, Ky_Gamma, f_Gamma) = \
            pde_obj_dict[pde].evaluate_grid(n_side_Omega=n, n_side_Gamma=n)
        true = u_Omega.reshape(n, n)
        ax = plt.Subplot(fig, inner[0])
        im = ax.imshow(true, cmap='Spectral', vmin=-.15, vmax=.15)
        ax.axis('off')
        ax.set_title('Truth', fontsize=12, y=.95)
        cax = fig.add_axes([0.06, .63, .035, .23])
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.tick_params(labelsize=12)
        fig.add_subplot(ax)

        #########
        # error #
        #########
        for i, m_type in enumerate(['vanilla', 'OLHP']):
            ax = plt.Subplot(fig, inner[i + 1])
            ax.imshow(pred_dict[f'{pde}_{m_type}'], cmap='Spectral', vmin=-.15,
                      vmax=.15)
            ax.axis('off')
            ax.set_title(m_type[0].upper() + m_type[1:], fontsize=12, y=.95)
            fig.add_subplot(ax)
        plt.savefig(f'results/{pde}.pdf', bbox_inches='tight')
