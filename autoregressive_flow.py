import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F

from utils import *


class LinearMasked(nn.Linear):
    """
    Class implementing nn.Linear with mask
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))

    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))

    def forward(self, x):
        return torch.nn.functional.linear(x, self.mask * self.weight, self.bias)


class MADE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.net = nn.Sequential(
            LinearMasked(input_dim, hidden_dim),
            nn.ReLU(),
            LinearMasked(hidden_dim, hidden_dim),
            nn.ReLU(),
            LinearMasked(hidden_dim, output_dim)
        )
        self.apply_masks()

    def forward(self, x):
        return self.net(x)

    def apply_masks(self):
        # Set order of masks, i.e. who can make which edges
        input_order = np.array([1, 2])
        output_order = np.concatenate((np.repeat(1, self.output_dim // 2), np.repeat(2, self.output_dim // 2)))
        hidden_order_1 = np.repeat(1, self.hidden_dim)
        hidden_order_2 = np.repeat(1, self.hidden_dim)

        # Create masks
        masks = []
        masks.append(input_order[:, None] <= hidden_order_1[None, :])
        masks.append(hidden_order_1[:, None] <= hidden_order_2[None, :])
        masks.append(hidden_order_2[:, None] < output_order[None, :])

        # Set the masks in all LinearMasked layers
        layers = [layer for layer in self.net.modules() if isinstance(layer, LinearMasked)]
        for layer, mask in zip(layers, masks):
            layer.set_mask(mask)


def train_ar_made_flow(train_data, test_data, dset_id):
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
    - a numpy array of size (n_train, 2) of floats in [0,1]^2. This represents
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
    # Using 5 gaussians, each with 3 variables; a weight π, mean µ and standard deviation σ.
    no_gaussians = 5
    input_dim = 2
    hidden_dim = 60
    output_dim = 2 * 3 * no_gaussians

    def get_zs_and_pzs(batch, output):
        x1_params = output[:, : 3 * no_gaussians]
        x2_params = output[:, 3 * no_gaussians:]

        x1_pis = F.softmax(x1_params[:, : no_gaussians], dim=1)
        x2_pis = F.softmax(x2_params[:, : no_gaussians], dim=1)

        x1_gaussians = dist.normal.Normal(loc=x1_params[:, no_gaussians: 2 * no_gaussians],
                                          scale=torch.exp(x1_params[:, 2 * no_gaussians:]))
        x2_gaussians = dist.normal.Normal(loc=x2_params[:, no_gaussians: 2 * no_gaussians],
                                          scale=torch.exp(x2_params[:, 2 * no_gaussians:]))

        cdf_1 = x1_gaussians.cdf(batch[:, 0].reshape(batch.shape[0], 1))
        cdf_2 = x2_gaussians.cdf(batch[:, 1].reshape(batch.shape[0], 1))

        z1 = torch.sum(cdf_1 * x1_pis, dim=1, keepdim=True)
        z2 = torch.sum(cdf_2 * x2_pis, dim=1, keepdim=True)

        p_z1 = torch.exp(x1_gaussians.log_prob(batch[:, 0].reshape(batch.shape[0], 1)))
        p_z2 = torch.exp(x2_gaussians.log_prob(batch[:, 1].reshape(batch.shape[0], 1)))

        dz1_dx1 = torch.abs(torch.sum(p_z1 * x1_pis, dim=1, keepdim=True))
        dz2_dx2 = torch.abs(torch.sum(p_z2 * x2_pis, dim=1, keepdim=True))

        return z1, z2, dz1_dx1, dz2_dx2

    def nll_loss(batch, output):
        z1, z2, dz1_dx1, dz2_dx2 = get_zs_and_pzs(batch, output)
        return -torch.mean(torch.log((dz1_dx1 * dz2_dx2) + 1e-8))

    # Model
    n_epochs = 400
    made = MADE(input_dim, hidden_dim, output_dim).cuda()
    optimizer = torch.optim.Adam(made.parameters())

    # Training
    train_losses = []
    test_losses = [nll_loss(test_data, made(test_data))]

    for epoch in range(n_epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            output = made(batch)
            loss = nll_loss(batch, output)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        print(f"[{100*(epoch+1)/n_epochs:.2f}%] Epoch {epoch + 1}") if (epoch + 1) % 20 == 0 else None
        test_loss = nll_loss(test_data, made(test_data))
        test_losses.append(test_loss.item())

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

    z1, z2, dz1_dx1, dz2_dx2 = get_zs_and_pzs(mesh_xs, made(mesh_xs))
    densities = dz1_dx1.detach().cpu() * dz2_dx2.detach().cpu()

    z1, z2, dz1_dx1, dz2_dx2 = get_zs_and_pzs(train_data, made(train_data))
    latents = np.column_stack((z1.detach().cpu(), z2.detach().cpu()))

    return train_losses, test_losses, densities, latents


def train_and_show_results_smiley():
    """
    Trains AR Flow MADE and displays samples and training plot for Smiley dataset
    """
    show_results_2d(1, train_ar_made_flow)


def train_and_show_results_half_moons():
    """
    Trains AR Flow MADE and displays samples and training plot for half moons dataset
    """
    show_results_2d(2, train_ar_made_flow)
