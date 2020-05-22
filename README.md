# Autoregressive Flow models and RealNVP in PyTorch
PyTorch implementations of Autoregressive Flows and RealNVP for both 2D and 3D datasets.

## Models

**Autoregressive Flow:**

In an autoregressive flow, we learn the transformation <img src="https://render.githubusercontent.com/render/math?math=z_i = f(x_i%3B x_{1:i-1})" style="display:inline; margin-bottom:-2px;">. The log-likelihood is then <img src="https://render.githubusercontent.com/render/math?math=\log p_\theta(x) = \sum_{i=1}^d \log p(z_i) %2B \log |\frac{dz_i}{dx_i}|" style="display:inline; margin-bottom:-2px;"> because the Jacobian is triangular. For each dimension, use the CDF of a mixture of Gaussians or Logistics to map to the latent space, which should be <img src="https://render.githubusercontent.com/render/math?math=z_i \sim \text{Unif}[0, 1]" style="display:inline; margin-bottom:-2px;"> i.i.d..

Fit <img src="https://render.githubusercontent.com/render/math?math=p_\theta" style="display:inline; margin-bottom:-2px;">  with maximum likelihood via stochastic gradient descent on the training set. Since this is a 2D problem, you can either learn <img src="https://render.githubusercontent.com/render/math?math=z_0 = f(x_0)" style="display:inline; margin-bottom:-2px;"> and <img src="https://render.githubusercontent.com/render/math?math=z_1 = f(x_1%3B x_0)" style="display:inline; margin-bottom:-2px;"> together with a MADE model, or with separate networks.

**RealNVP (2D):**

In this part, we want to train a flow with the following structure: 
<img src="https://render.githubusercontent.com/render/math?math=(z_1, z_2) = (f_{\theta, 1} \circ \cdots \circ f_{\theta, n})" style="display:inline; margin-bottom:-2px;">, where each <img src="https://render.githubusercontent.com/render/math?math=f_{\theta, i}" style="display:inline; margin-bottom:-2px;"> is an affine transformation of 1 dimension, conditioned on the other, and <img src="https://render.githubusercontent.com/render/math?math=z \sim N(0, I)" style="display:inline; margin-bottom:-2px;">. According to [Density Estimation Using Real NVP](https://arxiv.org/abs/1605.08803) Section 4.1, there's a particularly good way to parameterize the affine transformation:

Assuming that we're conditioning on <img src="https://render.githubusercontent.com/render/math?math=x_1" style="display:inline; margin-bottom:-2px;"> and transforming <img src="https://render.githubusercontent.com/render/math?math=x_2" style="display:inline; margin-bottom:-2px;">, we have 

- <img src="https://render.githubusercontent.com/render/math?math=z_1 = x_1" style="display:inline; margin-bottom:-2px;">

- <img src="https://render.githubusercontent.com/render/math?math=\text{log_scale} = \text{scale} \times tanh(g_{\theta, \text{scale}}(x_1)) %2B \text{scale_shift}" style="display:inline; margin-bottom:-2px;">

- <img src="https://render.githubusercontent.com/render/math?math=z_2 = exp(\text{log_scale}) \times x_2 %2B g_{\theta, \text{shift}}(x_1)" style="display:inline; margin-bottom:-2px;">

where <img src="https://render.githubusercontent.com/render/math?math=g_\theta" style="display:inline; margin-bottom:-2px;">, <img src="https://render.githubusercontent.com/render/math?math=\text{scale}" style="display:inline; margin-bottom:-2px;">, and <img src="https://render.githubusercontent.com/render/math?math=\text{scale_shift}" style="display:inline; margin-bottom:-2px;"> are all learned parameters.


<img src="https://render.githubusercontent.com/render/math?math=" style="display:inline; margin-bottom:-2px;">

**Autoregressive Flows using PixelCNN:**

Using the PixelCNN from previous [project](https://github.com/henrhoi/pixelcnn-pytorch) and using it as an autoregressive flow model on the black-and-white shapes dataset. Remember to dequantize the data and scale it between 0 and 1 for the autoregressive flow to have stable training.

**RealNVP (3D):**

Using the affine coupling flow from RealNVP and a form of [data-dependent initialization](https://arxiv.org/abs/1602.07868) that normalizes activations from an initial forward pass with a minibatch.


## Datasets

| Name | Dataset |
|------|---------|
| Smiley     |  ![](images/datasets/smiley.png)       |
| Half Moons     |  ![](images/datasets/half_moons.png)       |
| Shapes     |  ![](images/datasets/shapes.png)       |
| CelebA     |  ![](images/datasets/celeba.png)       |

## Results and samples

| Model | Dataset | Densities |  Latent space | 
|------|---------|---------|---------|
| AR Flow   | Smiley  |  ![](images/samples/smiley_densities_ar_flow.png)       | ![](images/samples/smiley_latent_space_ar_flow.png)|
| AR Flow  | Half Moons  | ![](images/samples/half_moons_densities_ar_flow.png)  | ![](images/samples/half_moons_latent_space_ar_flow.png) | 
| RealNVP (2D) | Smiley    |  ![](images/samples/smiley_densities_realnvp.png)       |![](images/samples/smiley_latent_space_realnvp.png) |
| RealNVP (2D) | Half Moons    |  ![](images/samples/half_moons_densities_realnvp.png)       |![](images/samples/half_moons_latent_space_realnvp.png) |




| Model | Dataset | Samples | Additional samples |
|------|---------|---------|---------|
| AR Flow w/ PixelCNN | Shapes    |  ![](images/samples/shapes_samples_ar_flow_pixelcnn.png)       |![](images/samples/shapes_samples_ar_flow_pixelcnn_floored.png) |
| RealNVP (3D) | CelebA    |  ![](images/samples/celeb_samples_realnvp.png)       | ![](images/samples/celeb_interpolations_realnvp.png) |

