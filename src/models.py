import torch.nn as nn
from einops import rearrange


class Autoencoder(nn.Module):
    """
    n-layer dense autoencoder for image data.
    
    Args:
        img_height (int): Height of the input image.
        img_width (int): Width of the input image.
        hidden_dim (int): Dimension of the hidden layers.
        latent_dim (int): Dimension of the latent (compressed) space.
        n_layers (int): Number of layers in the encoder and decoder networks.
        layernorm (bool): Whether to apply LayerNorm to the hidden layers (default: True).
    
    The autoencoder consists of `n_layers`, with the first and last layers 
    handling the transformation from input to hidden space, and hidden space 
    to latent space respectively. Intermediate layers apply ReLU activations 
    and optionally LayerNorm.
    """
    def __init__(
        self, img_height, img_width, hidden_dim, latent_dim, n_layers, layernorm=True
    ):
        super(Autoencoder, self).__init__()

        # Define input dimensions based on image height and width
        self.input_dim = img_height * img_width
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Ensure there are at least two layers (input->hidden, hidden->latent)
        assert n_layers >= 2, "n_layers must be >= 2."

        # Build the encoder and decoder layers
        self.encoder = []
        self.decoder = []
        for layer_ix in range(n_layers):
            if layer_ix == 0:  # add first layer
                self.encoder.append(nn.Linear(self.input_dim, self.hidden_dim))
                self.decoder.append(nn.Linear(self.latent_dim, self.hidden_dim))
            elif layer_ix == n_layers - 1:  # add last layer
                self.encoder.append(nn.Linear(self.hidden_dim, self.latent_dim))
                self.decoder.append(nn.Linear(self.hidden_dim, self.input_dim))
            else:
                self.encoder.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                self.decoder.append(nn.Linear(self.hidden_dim, self.hidden_dim))

            if (
                layernorm and layer_ix != n_layers - 1
            ):  # add layernorm if not the last layer
                self.encoder.append(nn.LayerNorm(self.hidden_dim))
                self.decoder.append(nn.LayerNorm(self.hidden_dim))

            if (
                layer_ix != n_layers - 1
            ):  # add activation function if not the last layer
                self.decoder.append(nn.ReLU())
                self.encoder.append(nn.ReLU())
            else:
                self.decoder.append(nn.Tanh())

        self.encoder = nn.Sequential(*self.encoder)
        self.decoder = nn.Sequential(*self.decoder)

    def forward(self, x):
        """
        Forward pass through the autoencoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            x_reconstructed (torch.Tensor): Reconstructed tensor of the same shape as the input.
        """
        # Get the batch size, channels, height, and width from input tensor
        b, c, h, w = x.shape

        # Flatten the image.
        x = rearrange(x, "b c h w -> b (c h w)")
        
        # Pass through the encoder and decoder to get latents and reconstructed tensors.
        latent = self.encoder(x)
        x_reconstructed = self.decoder(latent)

        # Reshape the reconstructed tensor back to image form
        x_reconstructed = rearrange(
            x_reconstructed, "b (c h w) -> b c h w", c=c, h=h, w=w
        )
        return x_reconstructed
