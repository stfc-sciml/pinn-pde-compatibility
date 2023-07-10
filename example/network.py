import sys
from time import time

import torch
import torch.nn as nn
from tqdm import trange


def activation(z, derivatives=False):
    """ activation function and its derivatives """
    sigma = torch.tanh(z)
    if derivatives:
        d_sigma = 1 - sigma ** 2
        dd_sigma = -2 * sigma * d_sigma
        return sigma, d_sigma, dd_sigma
    else:
        return sigma


class MembraneNet(nn.Module):
    """ PINN for membrane deformation """

    def __init__(self, hidden_units, OLHP=False, seed=0):
        """ constructor """
        torch.manual_seed(seed)
        super(MembraneNet, self).__init__()
        self.fcs = nn.ModuleList()
        n_in = 2
        for n_out in hidden_units:
            self.fcs.append(nn.Linear(n_in, n_out))
            n_in = n_out
        self.n_last_hidden = n_in
        self.OLHP = OLHP
        if OLHP:
            self.fc_out = nn.Linear(n_in - 1, 1)
            self.fc_out.weight.data.zero_()
        else:
            self.fc_out = nn.Linear(n_in, 1)

    def forward(self, xy, K, Kx, Ky, f):
        """ forward """
        # handle batch dim
        assert xy.ndim == 2
        batch_size = xy.size(0)

        # STEP 1 and 2 in Algorithm 1
        F = []
        s = []
        W = []  # also get the weights
        for fc in self.fcs:
            z = fc(xy)
            xy, d_sigma, dd_sigma = activation(z, derivatives=True)
            F.append(torch.diag_embed(d_sigma))
            s.append(dd_sigma)
            W.append(fc.weight)

        # STEP 3 in Algorithm 1 (P, Q)
        P = []
        Q = []
        for k, fc in enumerate(self.fcs):
            # P
            P_k = torch.stack(
                [torch.eye(self.n_last_hidden, device=xy.device)] * batch_size)
            for kk in range(len(self.fcs) - 1, k, -1):
                FW = torch.einsum('bij,jk->bik', F[kk], W[kk])
                P_k = torch.einsum('bij,bjk->bik', P_k, FW)
            P.append(P_k)
            # Q
            Q_k = torch.stack([W[k]] * batch_size)
            for kk in range(k - 1, -1, -1):
                FW = torch.einsum('bij,jk->bik', F[kk], W[kk])
                Q_k = torch.einsum('bij,bjk->bik', Q_k, FW)
            Q.append(Q_k)

        # STEP 4 in Algorithm 1 (U, V)
        F0Q0 = torch.einsum('bij,bjk->bik', F[0], Q[0])
        U = torch.einsum('bij,bjk->bik', P[0], F0Q0)
        V = torch.zeros(
            (batch_size, self.n_last_hidden, 2, 2), device=xy.device)
        for k, fc in enumerate(self.fcs):
            V += (P[k][:, :, :, None, None] * s[k][:, None, :, None, None] *
                  Q[k][:, None, :, :, None] * Q[k][:, None, :, None, :]
                  ).sum(dim=2)
        if not self.OLHP:
            # solution
            u = self.fc_out(xy).squeeze(dim=1)
            # PDE
            du = torch.einsum('j,bjm->bm', self.fc_out.weight[0], U)
            ux, uy = du[:, 0], du[:, 1]
            uxx = torch.einsum('j,bj->b', self.fc_out.weight[0], V[:, :, 0, 0])
            uyy = torch.einsum('j,bj->b', self.fc_out.weight[0], V[:, :, 1, 1])
            PDE = K * (uxx + uyy) + Kx * ux + Ky * uy + f
            return u, PDE
        else:
            # STEP 5 in Algorithm 1 (psi and g)
            # psi is determined by the target PDE
            psi = (K[:, None] * (V[:, :, 0, 0] + V[:, :, 1, 1]) +
                   Kx[:, None] * U[:, :, 0] +
                   Ky[:, None] * U[:, :, 1])

            # batch mean
            psi_batch_mean = psi.mean(dim=0)
            f_batch_mean = f.mean()

            # G
            G = psi_batch_mean[0] * torch.eye(self.n_last_hidden,
                                              device=xy.device)
            G[1:, 0] = -psi_batch_mean[1:]
            G[0, :] = -f_batch_mean * psi_batch_mean / (
                    psi_batch_mean ** 2).sum()

            # STEP 6 in Algorithm 1 (w)
            w = G[0, :].clone()
            w += (self.fc_out.weight[0][:, None] * G[1:, :]).sum(dim=0)

            # STEP 7 in Algorithm 1 (u)
            u = torch.einsum('i,bi->b', w, xy) + self.fc_out.bias[0]

            # pointwise PDE value
            du = torch.einsum('j,bjm->bm', w, U)
            ux, uy = du[:, 0], du[:, 1]
            uxx = torch.einsum('j,bj->b', w, V[:, :, 0, 0])
            uyy = torch.einsum('j,bj->b', w, V[:, :, 1, 1])
            PDE = K * (uxx + uyy) + Kx * ux + Ky * uy + f
            return u, PDE

    def predict_PDE(self, pde_obj, n_side,
                    device='cpu', batch_size=256, progress_bar=True):
        """ predict a PDE """
        # data
        (xy_Omega, u_Omega, K_Omega, Kx_Omega, Ky_Omega, f_Omega,
         xy_Gamma, u_Gamma, K_Gamma, Kx_Gamma, Ky_Gamma, f_Gamma) \
            = pde_obj.evaluate_grid(n_side_Omega=n_side,
                                    n_side_Gamma=n_side, device=device)

        # model state
        self.eval()
        self.to(device)

        # Omega
        u_Omega_pred = torch.zeros_like(u_Omega)
        PDE_Omega_pred = torch.zeros_like(u_Omega)
        batches = trange(0, len(xy_Omega), batch_size,
                         desc='Predict Omega', unit='batch',
                         disable=not progress_bar, ascii=True, leave=True,
                         file=sys.stdout)
        with torch.no_grad():
            for start in batches:
                bat = slice(start, start + batch_size)
                u_Omega_pred[bat], PDE_Omega_pred[bat] = self.forward(
                    xy_Omega[bat], K_Omega[bat],
                    Kx_Omega[bat], Ky_Omega[bat], f_Omega[bat])

        # Gamma
        u_Gamma_pred = torch.empty_like(u_Gamma)
        PDE_Gamma_pred = torch.empty_like(u_Gamma)
        batches = trange(0, len(xy_Gamma), batch_size,
                         desc='Predict Gamma', unit='batch',
                         disable=not progress_bar, ascii=True, leave=True,
                         file=sys.stdout)
        with torch.no_grad():
            for start in batches:
                bat = slice(start, start + batch_size)
                u_Gamma_pred[bat], PDE_Gamma_pred[bat] = self.forward(
                    xy_Gamma[bat], K_Gamma[bat],
                    Kx_Gamma[bat], Ky_Gamma[bat], f_Gamma[bat])

        # error
        loss_func = nn.MSELoss(reduction='mean')
        L_Omega = loss_func(u_Omega_pred, u_Omega).item()
        L_Gamma = loss_func(u_Gamma_pred, u_Gamma).item()
        L_PDE = (PDE_Omega_pred ** 2).mean().item()
        return (u_Omega_pred, u_Gamma_pred, PDE_Omega_pred,
                L_Omega, L_Gamma, L_PDE)

    def train_PDE(self, pde_obj, n_side_train, n_side_test,
                  beta_Omega=1., beta_Gamma=1., beta_PDE=1.,
                  device='cpu', batch_size=256, lr=1e-3, epochs=10,
                  test_interval=10, progress_bar=True,
                  save_dir=None, checkpoint_interval=10000):
        """ train a PDE """
        # data
        (xy_Omega, u_Omega, K_Omega,
         Kx_Omega, Ky_Omega, f_Omega,
         xy_Gamma, u_Gamma, K_Gamma,
         Kx_Gamma, Ky_Gamma, f_Gamma) \
            = pde_obj.evaluate_grid(n_side_Omega=n_side_train,
                                    n_side_Gamma=0, device=device)

        # optimizer and loss functions
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_func = nn.MSELoss(reduction='mean')

        # training history
        hist_keys = ['wct', 'L_Omega', 'L_Gamma', 'L_PDE']
        hist = {key: [] for key in hist_keys}

        # model state
        self.to(device)

        # timing
        t0 = time()

        # epoch loop
        for epoch in range(epochs):
            # shuffle data for each epoch
            gen = torch.Generator()
            gen = gen.manual_seed(epoch)
            perm = torch.randperm(len(xy_Omega), generator=gen)
            xy_Omega, u_Omega, K_Omega, Kx_Omega, Ky_Omega, f_Omega = \
                (xy_Omega[perm], u_Omega[perm], K_Omega[perm],
                 Kx_Omega[perm], Ky_Omega[perm], f_Omega[perm])
            xy_Gamma, u_Gamma, K_Gamma, Kx_Gamma, Ky_Gamma, f_Gamma = \
                (xy_Gamma[perm], u_Gamma[perm], K_Gamma[perm],
                 Kx_Gamma[perm], Ky_Gamma[perm], f_Gamma[perm])

            ############################
            # training on mini-batches #
            ############################
            batches = trange(0, len(xy_Omega), batch_size,
                             desc=f'Epoch {epoch + 1}   Train', unit='batch',
                             disable=not progress_bar, ascii=True, leave=True,
                             file=sys.stdout)
            for start in batches:
                # restore training after testing
                self.train()

                # zero grad for each batch
                optimizer.zero_grad()

                # batch
                bat = slice(start, start + batch_size)

                # Omega
                u_Omega_pred_bat, PDE_Omega_pred_bat = self.forward(
                    xy_Omega[bat], K_Omega[bat],
                    Kx_Omega[bat], Ky_Omega[bat], f_Omega[bat])
                L_Omega = loss_func(u_Omega_pred_bat, u_Omega[bat])
                L_PDE = (PDE_Omega_pred_bat ** 2).mean()

                # Gamma
                u_Gamma_pred_bat, _ = self.forward(
                    xy_Gamma[bat], K_Gamma[bat],
                    Kx_Gamma[bat], Ky_Gamma[bat], f_Gamma[bat])
                L_Gamma = loss_func(u_Gamma_pred_bat, u_Omega[bat])

                # loss
                L = (beta_Omega * L_Omega + beta_Gamma * L_Gamma
                     + beta_PDE * L_PDE)

                # backprop
                L.backward()
                optimizer.step()
                batches.set_postfix_str(f'L={L.item():.2e}')

            ############################
            # testing after each epoch #
            ############################
            if (epoch + 1) % test_interval == 0:
                _, _, _, L_Omega, L_Gamma, L_PDE = \
                    self.predict_PDE(pde_obj, n_side=n_side_test,
                                     device=device, batch_size=batch_size,
                                     progress_bar=progress_bar)
                # verbose
                wct = torch.tensor(time() - t0)
                t0 = time()
                if progress_bar:
                    line = f'[Epoch {epoch + 1}] Losses: '
                    for key in hist_keys:
                        if key != 'wct':
                            fmt = 's' if eval(key) == 'NA' else '.2e'
                            line += f'{key}: {eval(key):{fmt}}; '
                    line += f'wct: {wct:.1f} sec'
                    print(line, flush=True)

                # history
                for key in hist_keys:
                    hist[key].append(eval(key))

            # save results
            if save_dir is not None:
                if (epoch + 1) % checkpoint_interval == 0 or \
                        epoch + 1 == epochs:
                    torch.save(hist, str(save_dir) +
                               f'/training_history__epoch={epoch + 1}.pt')
                    torch.save(self.state_dict(), str(save_dir) +
                               f'/model_weights__epoch={epoch + 1}.pt')
        return hist
