import torch.nn as nn

from utils import *


class MaskedConv2d(nn.Conv2d):
    """
    Class extending nn.Conv2d to use masks.
    """

    def __init__(self, mask_type, in_channels, out_channels, kernel_size, padding=0):
        super(MaskedConv2d, self).__init__(in_channels, out_channels, kernel_size, padding=padding)
        self.register_buffer('mask', torch.ones(out_channels, in_channels, kernel_size, kernel_size).float())

        # _, depth, height, width = self.weight.size()
        h, w = kernel_size, kernel_size

        if mask_type == 'A':
            self.mask[:, :, h // 2, w // 2:] = 0
            self.mask[:, :, h // 2 + 1:, :] = 0
        else:
            self.mask[:, :, h // 2, w // 2 + 1:] = 0
            self.mask[:, :, h // 2 + 1:, :] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


class AutoregressiveFlowPixelCNN(nn.Module):
    """
    Autoregressive Flow PixelCNN-class
    """

    def __init__(self, in_channels, conv_filters, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            # A 7x7 A-type convolution
            MaskedConv2d('A', in_channels=in_channels, out_channels=conv_filters, kernel_size=7, padding=3),
            nn.BatchNorm2d(conv_filters),
            nn.ReLU(),
            # 5 7x7 B-type convolutions
            MaskedConv2d('B', in_channels=conv_filters, out_channels=conv_filters, kernel_size=7, padding=3),
            nn.BatchNorm2d(conv_filters),
            nn.ReLU(),
            MaskedConv2d('B', in_channels=conv_filters, out_channels=conv_filters, kernel_size=7, padding=3),
            nn.BatchNorm2d(conv_filters),
            nn.ReLU(),
            MaskedConv2d('B', in_channels=conv_filters, out_channels=conv_filters, kernel_size=7, padding=3),
            nn.BatchNorm2d(conv_filters),
            nn.ReLU(),
            MaskedConv2d('B', in_channels=conv_filters, out_channels=conv_filters, kernel_size=7, padding=3),
            nn.BatchNorm2d(conv_filters),
            nn.ReLU(),
            MaskedConv2d('B', in_channels=conv_filters, out_channels=conv_filters, kernel_size=7, padding=3),
            nn.BatchNorm2d(conv_filters),
            nn.ReLU(),
            # 2 1x1 B-type convolutions
            MaskedConv2d('B', in_channels=conv_filters, out_channels=conv_filters, kernel_size=1),
            nn.BatchNorm2d(conv_filters),
            nn.ReLU(),
            MaskedConv2d('B', in_channels=conv_filters, out_channels=out_channels, kernel_size=1)).cuda()

    def forward(self, x):
        return self.net(x)


def train_autoregressive_flow_pixelcnn(train_data, test_data):
    """
    train_data: A (n_train, H, W, 1) uint8 numpy array of binary images with values in {0, 1}
    test_data: A (n_test, H, W, 1) uint8 numpy array of binary images with values in {0, 1}
    H = W = 20
    Note that you should dequantize your train and test data, your dequantized pixels should all lie in [0,1]

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (100, H, W, 1) of samples with values in [0, 1], where [0,0.5] represents a black pixel
        and [0.5,1] represents a white pixel. We will show your samples with and without noise.
    """

    def dequantize(mat):
        """
        Dequantizes and normalizes between [0, 1]
        """
        return torch.from_numpy((mat + np.random.uniform(low=-.5, high=.5, size=mat.shape) + .5) / 2).float().cuda()

    train_data = dequantize(np.transpose(train_data, [0, 3, 1, 2]))
    test_data = dequantize(np.transpose(test_data, [0, 3, 1, 2]))

    def nll_loss(batch, output):
        # Output shape (N, C, H, W) --> C = 3 * no_gaussians
        pis = torch.softmax(output[:, :no_gaussians], dim=1)
        means = output[:, no_gaussians: 2 * no_gaussians]
        stds = torch.exp(output[:, -no_gaussians:])

        gaussians = torch.distributions.normal.Normal(loc=means, scale=stds)  # Shape is (N, no_gaussians, H, W)

        # Batch shape is (N, 1, H, W), needs to be in (N, no_gaussians, H, W) for broadcasting
        batch_reshaped = torch.cat(tuple([batch] * no_gaussians), dim=1)  # NB: Broadcasting

        cdf = gaussians.cdf(batch_reshaped)
        z = torch.sum(cdf * pis, dim=1, keepdim=True)
        dz_dx_normals = torch.exp(gaussians.log_prob(
            batch_reshaped)) / 2  # dzdz = differentiated integral - Shape (N, no_gaussians, H, W), dividing by 2 because of the dequantization. NB: Broadcasting
        dz_dx = torch.abs(
            torch.sum(dz_dx_normals * pis, dim=1, keepdim=True))  # Shape is (N, 1, H, W) - Called Jacobian

        summed_dz_dx = torch.sum(torch.log(dz_dx + 1e-8).view(batch.shape[0], -1), dim=1,
                                 keepdim=False)  # Shape is (N, 1)
        return -torch.mean(summed_dz_dx)

    def get_batched_loss(data_loader, model):
        test_loss = []
        for batch in data_loader:
            out = model(batch)
            loss = nll_loss(batch, out)
            test_loss.append(loss.item())

        return np.mean(np.array(test_loss))

    dataset_params = {
        'batch_size': 128,
        'shuffle': True
    }

    n_epochs = 30
    lr = 0.005

    # Using 5 gaussians, each with 3 variables; a weight π, mean µ and standard deviation σ.
    no_gaussians = 5
    no_channels, convolution_filters = 1, 64

    pixelcnn = AutoregressiveFlowPixelCNN(no_channels, convolution_filters, 3 * no_gaussians).cuda()
    train_loader = torch.utils.data.DataLoader(train_data, **dataset_params)
    test_loader = torch.utils.data.DataLoader(test_data, **dataset_params)

    optimizer = torch.optim.Adam(pixelcnn.parameters(), lr=lr)

    train_losses = []
    test_losses = [get_batched_loss(test_loader, pixelcnn)]

    for epoch in range(n_epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            output = pixelcnn(batch)
            loss = nll_loss(batch, output)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        test_loss = get_batched_loss(test_loader, pixelcnn)
        test_losses.append(test_loss)
        print(f"[{100*(epoch+1)/n_epochs:.2f}%] Epoch {epoch + 1}") if (epoch + 1) % 5 == 0 else None

    def get_x_samples(samples, model):
        nonlocal no_gaussians

        output = model(samples)
        # Output shape (N, C, H, W) --> C = 3 * no_gaussians
        uniform_zs = torch.zeros_like(samples).uniform_(0, 1)  # Gets a uniformed distributed z's in [0, 1]

        pis = torch.softmax(output[:, :no_gaussians], dim=1)
        means = output[:, no_gaussians: 2 * no_gaussians]
        stds = torch.exp(output[:, -no_gaussians:])

        gaussians = torch.distributions.normal.Normal(loc=means, scale=stds)  # Shape is (N, no_gaussians, H, W)

        # Batch shape is (N, 1, H, W), needs to be in (N, no_gaussians, H, W) for broadcasting
        z_reshaped = torch.cat(tuple([uniform_zs] * no_gaussians), dim=1)  # NB: Broadcasting

        xs = gaussians.icdf(z_reshaped)

        # x = torch.sum(icdf * pis, dim=1, keepdim=True)
        return xs, pis

    torch.cuda.empty_cache()
    del train_data, test_data

    H, W = 20, 20
    samples = torch.zeros(size=(100, 1, H, W)).cuda()

    pixelcnn.eval()
    with torch.no_grad():
        for i in range(H):
            for j in range(W):
                xs, pis = get_x_samples(samples, pixelcnn)
                classes = torch.multinomial(pis[:, :, i, j], 1)
                samples[:, :, i, j] = torch.gather(input=xs[:, :, i, j], dim=1, index=classes)

    return np.array(train_losses) / (H * W), \
           np.array(test_losses) / (H * W), \
           np.transpose(samples.detach().cpu().numpy(), [0, 2, 3, 1])


def train_and_show_results_shapes():
    """
    Trains Autoregressive Flow PixelCNN and displays samples and training plot for Shapes dataset
    """
    show_results_shapes(train_autoregressive_flow_pixelcnn)


