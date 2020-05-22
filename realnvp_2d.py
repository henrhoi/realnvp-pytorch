import torch.nn as nn

from utils import *


class NPVNode(nn.Module):
    def __init__(self):
        super(NPVNode, self).__init__()
        self.scale = nn.Linear(in_features=1, out_features=1)  # Scale is Ax + b, i.e. affine transformation

        self.g_theta_scale = nn.Sequential(nn.Linear(in_features=1, out_features=32), nn.LeakyReLU(),
                                           nn.Linear(in_features=32, out_features=16), nn.LeakyReLU(),
                                           nn.Linear(in_features=16, out_features=1))

        self.g_theta_shift = nn.Sequential(nn.Linear(in_features=1, out_features=32), nn.LeakyReLU(),
                                           nn.Linear(in_features=32, out_features=16), nn.LeakyReLU(),
                                           nn.Linear(in_features=16, out_features=1))

    def forward(self, x1, x2, prev_log_determinant):
        x1, x2 = x2, x1

        z1 = x1
        log_scale = self.scale(torch.tanh(self.g_theta_scale(x1)))
        z2 = torch.exp(log_scale) * x2 + self.g_theta_shift(x1)

        current_log_determinant = log_scale + prev_log_determinant
        return z1, z2, current_log_determinant


class RealNVP(nn.Module):
    def __init__(self, num_layers=15):
        super(RealNVP, self).__init__()
        modules = [NPVNode() for _ in range(num_layers)]
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        z1, z2 = x[:, [1]], x[:, [0]]
        log_determinant = torch.tensor(0.)
        for layer in self.net:
            z1, z2, log_determinant = layer(z1, z2, log_determinant)

        return torch.cat((z1, z2), dim=1), log_determinant


def train_realnvp_2d(train_data, test_data, dset_id):
    """
    train_data: An (n_train, 2) numpy array of floats in R^2
    test_data: An (n_test, 2) numpy array of floats in R^2
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
                used to set different hyperparameters for different datasets, or
                for plotting a different region of densities

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (?,) of probabilities with values in [0, +infinity).
        Refer to the commented hint.
    - a numpy array of size (n_train, 2) of floats in R^2. This represents
        mapping the train set data points through our flow to the latent space.
    """
    # Create data loaders
    dataset_params = {
        'batch_size': 128,
        'shuffle': False
    }

    train_data = torch.from_numpy(train_data).float().cuda()
    test_data = torch.from_numpy(test_data).float().cuda()

    train_loader = torch.utils.data.DataLoader(train_data, **dataset_params)

    def nll_loss(batch, output):
        z, log_determinant = output
        gaussian = torch.distributions.normal.Normal(loc=0, scale=1)

        # p_theta
        log_pz = torch.sum(gaussian.log_prob(z), dim=1, keepdim=True)
        p_theta = log_pz + log_determinant
        return -torch.mean(p_theta), torch.exp(p_theta)  # Last return is for density plot

    # Model
    n_epochs = 300
    realnvp = RealNVP().cuda()
    optimizer = torch.optim.Adam(realnvp.parameters())

    # Training
    train_losses = []
    test_losses = [nll_loss(test_data, realnvp(test_data))[0].item()]

    for epoch in range(n_epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            output = realnvp(batch)
            loss = nll_loss(batch, output)[0]
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        test_loss = nll_loss(test_data, realnvp(test_data))[0]
        test_losses.append(test_loss.item())
        if (epoch + 1) % 20 == 0:
            print(f"[{100*(epoch+1)/n_epochs:.2f}%] Epoch {epoch + 1} - Test loss: {test_loss:.2f}")

    # Creating heatmap
    dx, dy = 0.025, 0.025
    if dset_id == 1:  # face
        x_lim = (-4, 4)
        y_lim = (-4, 4)
    elif dset_id == 2:  # two moons
        x_lim = (-1.5, 2.5)
        y_lim = (-1, 1.5)
    y, x = np.mgrid[slice(y_lim[0], y_lim[1] + dy, dy), slice(x_lim[0], x_lim[1] + dx, dx)]
    mesh_xs = torch.FloatTensor(np.stack([x, y], axis=2).reshape(-1, 2)).cuda()

    densities = nll_loss(mesh_xs, realnvp(mesh_xs))[1].detach().cpu().numpy()
    latents = realnvp(train_data)[0].detach().cpu().numpy()
    return train_losses, test_losses, densities, latents


def train_and_show_results_smiley():
    """
    Trains RealNVP and displays samples and training plot for Smiley dataset
    """
    show_results_2d(1, train_realnvp_2d)


def train_and_show_results_half_moons():
    """
    Trains RealNVP and displays samples and training plot for half moons dataset
    """
    show_results_2d(2, train_realnvp_2d)
