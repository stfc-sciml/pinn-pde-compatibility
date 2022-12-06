import torch
import torch.nn as nn


# Target PDE:
# u_xx + u_yy - u_tt - u_t = f
# where f = sin(x + y - t)

def f_func(x):
    """ source term """
    return torch.sin(x[:, 0] + x[:, 1] - x[:, 2])


def activation_func(z, derivatives=False):
    """ activation function and its derivatives """
    sigma = torch.tanh(z)
    if derivatives:
        d_sigma = 1 - sigma ** 2
        dd_sigma = -2 * sigma * d_sigma
        return sigma, d_sigma, dd_sigma
    else:
        return sigma


def mm_batch(A1, A2):
    """ matrix multiplication considering batch dim """
    if A1.ndim == 3 and A2.ndim == 3:
        return torch.einsum('bij,bjk->bik', A1, A2)
    elif A1.ndim == 2 and A2.ndim == 3:
        return torch.einsum('ij,bjk->bik', A1, A2)
    elif A1.ndim == 3 and A2.ndim == 2:
        return torch.einsum('bij,jk->bik', A1, A2)
    elif A1.ndim == 2 and A2.ndim == 2:
        return torch.mm(A1, A2)
    else:
        assert False


class Net(nn.Module):
    def __init__(self, n_input=3, n_last_hidden=4):
        super(Net, self).__init__()
        self.n_input = n_input
        self.n_last_hidden = n_last_hidden

        self.fc = nn.ModuleList()
        self.fc.append(nn.Linear(n_input, 4))
        self.fc.append(nn.Linear(4, 8))
        self.fc.append(nn.Linear(8, n_last_hidden))

        # we need (n_last_hidden-1) lambda's
        # Note: this is implemented as a linear layer so that PyTorch can
        #       handle many things for us, such as backprop, device transfer,
        #       initialization and model saving.
        self.lmbda = nn.Linear(n_last_hidden - 1, 1, bias=False)

    def forward(self, x):
        # handle batch dim
        assert x.ndim == 2
        batch_size = x.size(0)

        # STEP 1 in Algorithm 1
        # Note: before propagation, make a copy of x0 requiring no gradient
        #       because we have considered its contribution to J and H
        #       analytically via U and V
        x_no_grad = x.clone().detach()
        L = len(self.fc) + 1
        for k in range(L - 1):
            # with gradient to W, with gradient to x0
            x = activation_func(self.fc[k](x))

        # compute f before x_no_grad is processed by layers
        f = f_func(x_no_grad)

        # STEP 2 in Algorithm 1 (F, s)
        F = []
        s = []
        W = []  # also get the weights
        for k in range(L - 1):
            # with gradient to W, without gradient to x0
            z = self.fc[k](x_no_grad)
            x_no_grad, d_sigma, dd_sigma = activation_func(z,
                                                           derivatives=True)
            # TODO: don't know how to use diag() excluding the batch dim,
            #       so using a loop
            F_k = []
            for i in range(batch_size):
                F_k.append(torch.diag(d_sigma[i]))
            F.append(torch.stack(F_k))
            s.append(dd_sigma)
            W.append(self.fc[k].state_dict()['weight'])

        # STEP 3 in Algorithm 1 (P, Q)
        P = []
        Q = []
        for k in range(L - 1):
            # P
            P_k = torch.stack([torch.eye(self.n_last_hidden, device=x.device)] *
                              batch_size)
            for kk in range(L - 2, k, -1):
                P_k = mm_batch(P_k, mm_batch(F[kk], W[kk]))
            P.append(P_k)
            # Q
            Q_k = torch.stack([W[k]] * batch_size)
            for kk in range(k - 1, -1, -1):
                Q_k = mm_batch(Q_k, mm_batch(F[kk], W[kk]))
            Q.append(Q_k)

        # STEP 4 in Algorithm 1 (U, V)
        U = mm_batch(P[0], mm_batch(F[0], Q[0]))
        V = torch.zeros(
            (batch_size, self.n_last_hidden, self.n_input, self.n_input),
            device=x.device)
        for k in range(0, L - 1):
            V += (P[k][:, :, :, None, None] * s[k][:, None, :, None, None] *
                  Q[k][:, None, :, :, None] * Q[k][:, None, :, None, :]
                  ).sum(dim=2)

        # STEP 5 in Algorithm 1 (psi and g)
        # this is determined by the target PDE
        # H[0, 0] + H[1, 1] - H[2, 2] - J[2]
        psi = V[:, :, 0, 0] + V[:, :, 1, 1] - V[:, :, 2, 2] - U[:, :, 2]
        G = torch.stack(
            [torch.eye(self.n_last_hidden, device=x.device)] * batch_size)
        G[:, 0, 0] = f[:] / psi[:, 0]
        G[:, 1:, 0] = -psi[:, 1:] / psi[:, 0, None]

        # STEP 6 in Algorithm 1 (w)
        w = G[:, 0, :]
        lmbda = self.lmbda.state_dict()['weight'][0]
        for p in range(1, self.n_last_hidden):
            w += lmbda[p - 1] * G[:, p, :]

        # STEP 7 in Algorithm 1 (u)
        u_out = torch.einsum('bi,bi->b', w, x)
        return u_out


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    # model
    n_in = 3
    model = Net(n_input=n_in, n_last_hidden=4)
    # input
    n_batch = 4
    x0 = torch.rand((n_batch, n_in), requires_grad=True)
    # u, Jacobian, Hessian, f
    u = model.forward(x0)
    Js = []
    Hs = []
    for i_batch in range(n_batch):
        J = torch.autograd.functional.jacobian(
            model.forward, x0[i_batch].unsqueeze(0))[0][0]
        H = torch.autograd.functional.hessian(
            model.forward, x0[i_batch].unsqueeze(0))[0, :, 0, :]
        Js.append(J)
        Hs.append(H)
    fx0 = f_func(x0)

    # report results
    for i_batch in range(n_batch):
        print(f'============ DATA POINT {i_batch} ============')
        print(f'**** INPUT ****')
        print(f'x0 = {x0[i_batch].detach().numpy()}')
        print(f'**** OUTPUT ****')
        print(f'u = {u[i_batch].item()}')
        J = Js[i_batch]
        H = Hs[i_batch]
        print(f'Jacobian = {J.detach().numpy()}')
        print(f'Hessian =\n{H.detach().numpy()}')
        print(f'**** PDE ****')
        print('Equation: u_xx + u_yy - u_tt - u_t - f = 0')
        print(f'MLP approximated values:')
        print(f'    u_xx = H[0, 0] = {H[0, 0]}')
        print(f'    u_yy = H[1, 1] = {H[1, 1]}')
        print(f'    u_tt = H[2, 2] = {H[2, 2]}')
        print(f'    u_t = J[2] = {J[2]}')
        print(f'    f = {fx0[i_batch]}')
        print(f'    u_xx + u_yy - u_tt - u_t - f = '
              f'{H[0, 0] + H[1, 1] - H[2, 2] - J[2] - fx0[i_batch]}')
        print('\n')
    print(f'============ CONCLUSION ============')
    print('An MLP equipped with the out-layer-hyperplane can "'
          'enforce" the PDE to hold.')
