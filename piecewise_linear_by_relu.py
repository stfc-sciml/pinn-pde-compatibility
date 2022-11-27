import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, n_input):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_input, 10, bias=True)
        self.fc2 = nn.Linear(10, 1, bias=False)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    # input
    x0 = torch.linspace(-1, 1, 100).unsqueeze(1)
    # output
    u_list = []
    # create different models by changing seed
    for seed in range(10):
        torch.manual_seed(seed)
        # model
        model = Net(1)
        # pass
        with torch.no_grad():
            u = model.forward(x0)
        u_list.append(u)

    # plot
    plt.figure(dpi=100)
    for i, u in enumerate(u_list):
        plt.plot(x0[:, 0].detach(), u[:, 0].detach(), label=f'M{i}')
        plt.xlabel('x')
        plt.ylabel('u')
    plt.legend()
    plt.show()
