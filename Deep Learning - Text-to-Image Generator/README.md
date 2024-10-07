# Text to Image Generation App

This is a Streamlit application that generates images from text prompts using the Stable Diffusion model.

## Requirements

- Python 3.11
- `requirements.txt` contains the necessary packages to install.

## Mathematical Foundation

- Diffusion Process
     - Forward (Noise Addition): Progressively adds noise to an image.
       - $$q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{\alpha_t}x_{t-1}, (1 - \alpha_t)I)$$
     - Reverse (Denoising): Learns to remove noise and recover the image.
       - $$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t)$$
         
- Latent Space with Variational Auto-Encoders (VAE)
  - Compresses images into latent space via a Variational Autoencoder (VAE), enabling efficient diffusion.
     - $$z \sim q_\phi(z|x)$$
       
- U-Net Architecture
   - Uses a U-Net (CNN) to predict and remove noise at each step, leveraging skip connections for detail preservation.
     
- Text Conditioning
   - Conditions the image generation process using text embeddings from a Transformer-based model.
 
- Optimization
   - Trains the model by minimizing the mean squared error (MSE) between predicted and actual noise.
     - $$L_\theta=E_{x_0, \epsilon, t} [||\epsilon - \epsilon_\theta(x_t, t)||^2]$$
