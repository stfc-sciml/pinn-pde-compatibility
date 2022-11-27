import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, n_input):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_input, 4)
        self.fc2 = nn.Linear(4, 8)
        self.fc3 = nn.Linear(8, 4)
        self.fc4 = nn.Linear(4, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        return x


if __name__ == "__main__":
    # model
    n_in = 2
    model = Net(n_in)
    # input
    x0 = torch.rand(n_in, requires_grad=True)
    # u, Jacobian, Hessian
    u = model.forward(x0)
    J = torch.autograd.functional.jacobian(model.forward, x0)[0]
    H = torch.autograd.functional.hessian(model.forward, x0)
    # report results
    print(f'**** INPUT ****')
    print(f'x0 = {x0.detach().numpy()}')
    print(f'\n**** OUTPUT ****')
    print(f'u = {u.item()}')
    print(f'Jacobian = {J.detach().numpy()}')
    print(f'Hessian =\n{H.detach().numpy()}')
    print(f'\n**** CONCLUSION ****')
    print('A ReLU-based MLP will always cause a vanished Hessian.')
