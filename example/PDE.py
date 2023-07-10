import numpy as np
import torch
from sympy import *


class MembranePDE:
    """ sympy-based PDE object for membrane """

    def __init__(self, u_str, K_str):
        """ constructor """
        # variables
        x, y = Symbol('x'), Symbol('y')
        # u
        self.u = eval(u_str)
        # K and derivatives
        self.K = eval(K_str)
        self.Kx = diff(self.K, x)
        self.Ky = diff(self.K, y)
        # f
        u_x, u_y = diff(self.u, x), diff(self.u, y)
        u_xx, u_yy = diff(u_x, x), diff(u_y, y)
        self.f = -(self.K * (u_xx + u_yy) +
                   self.Kx * u_x + self.Ky * u_y)

    def evaluate(self, xy, device='cpu'):
        """ evaluate u, K, Kx, Ky, f at given locations """

        def eval_to_tensor(func):
            x, y = Symbol('x'), Symbol('y')
            val = lambdify([x, y], func, 'numpy')(xy[:, 0], xy[:, 1])
            return torch.from_numpy(val).to(torch.float).to(device)

        u = eval_to_tensor(self.u)
        K = eval_to_tensor(self.K)
        Kx = eval_to_tensor(self.Kx)
        Ky = eval_to_tensor(self.Ky)
        f = eval_to_tensor(self.f)
        return u, K, Kx, Ky, f

    def evaluate_grid(self, n_side_Omega, n_side_Gamma=0, device='cpu'):
        """ evaluate u, K, Kx, Ky, f on a grid """
        # Omega
        crd = np.linspace(0, 1, n_side_Omega)
        x, y = np.meshgrid(crd, crd, indexing='xy')
        xy_Omega = np.array([x.reshape(-1), y.reshape(-1)]).T
        u_Omega, K_Omega, Kx_Omega, Ky_Omega, f_Omega = \
            self.evaluate(xy_Omega, device=device)
        xy_Omega = torch.from_numpy(xy_Omega).to(torch.float).to(device)

        # Gamma
        if n_side_Gamma == 0:
            # use the same amount
            assert n_side_Omega % 2 == 0
            n_side_Gamma = n_side_Omega ** 2 // 4
        crd = np.linspace(0, 1, n_side_Gamma)
        zero = np.zeros_like(crd)
        one = np.ones_like(crd)
        x_Gamma = np.concatenate([zero, one, crd, crd], axis=0)
        y_Gamma = np.concatenate([crd, crd, zero, one], axis=0)
        xy_Gamma = np.array([x_Gamma, y_Gamma]).T
        u_Gamma, K_Gamma, Kx_Gamma, Ky_Gamma, f_Gamma = \
            self.evaluate(xy_Gamma, device=device)
        xy_Gamma = torch.from_numpy(xy_Gamma).to(torch.float).to(device)
        return (xy_Omega, u_Omega, K_Omega, Kx_Omega, Ky_Omega, f_Omega,
                xy_Gamma, u_Gamma, K_Gamma, Kx_Gamma, Ky_Gamma, f_Gamma)


# PDEs used in this experiment
pde_obj_dict = {
    'smooth': MembranePDE(
        u_str='x * (x - 1) * sin(2 * pi * y) * cos(x * 2 - y * x + x)',
        K_str='1 + cos(x * 4 + y * 8) / 4'),
    'sharp': MembranePDE(
        u_str='x * (x - 1) * sin(2 * pi * y) * ('
              'exp(-((x - .7) ** 2 + (y - .7) ** 2) / .005) '
              '+ exp(-((x - .3) ** 2 + (y - .3) ** 2) / .005))',
        K_str='1 + cos(x * 10 + y * 10) / 4')
}
