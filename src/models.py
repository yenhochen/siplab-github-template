import torch
import torch.nn as nn
from einops import rearrange


class Autoencoder(nn.Module):
    """
    n-layer dense autoencoder
    """
    def __init__(self, 
                img_height, 
                img_width,
                hidden_dim,
                latent_dim,
                n_layers, 
                layernorm=True):
        super(Autoencoder, self).__init__()

        self.input_dim = img_height*img_width
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        assert n_layers >= 2, "n_layers must be >= 2."

        self.encoder = []
        self.decoder = []
        for layer_ix in range(n_layers):
            if layer_ix == 0: # add first layer
                self.encoder.append(nn.Linear(self.input_dim, self.hidden_dim))
                self.decoder.append(nn.Linear(self.latent_dim, self.hidden_dim))
            elif layer_ix == n_layers - 1: # add last layer
                self.encoder.append(nn.Linear(self.hidden_dim, self.latent_dim))
                self.decoder.append(nn.Linear(self.hidden_dim, self.input_dim))
            else:
                self.encoder.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                self.decoder.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            
            if layernorm and layer_ix != n_layers - 1: # add layernorm if not the last layer
                self.encoder.append(nn.LayerNorm(self.hidden_dim))
                self.decoder.append(nn.LayerNorm(self.hidden_dim))

            
            if layer_ix != n_layers - 1: # add activation function if not the last layer
                self.decoder.append(nn.ReLU())
                self.encoder.append(nn.ReLU())
            else:
                self.decoder.append(nn.Tanh())

        self.encoder = nn.Sequential(*self.encoder)
        self.decoder = nn.Sequential(*self.decoder)

    def forward(self, x):
        """
        x: torch.Tensor (b, c, h, w) for batch, channel, height, width
        returns:
            x_reconstructed: torch.Tensor (b, c, h, w) for batch, channel, height, width
        """
        b, c, h, w = x.shape
        x = rearrange(x, "b c h w -> b (c h w)")
        latent = self.encoder(x)
        x_reconstructed = self.decoder(latent)
        x_reconstructed = rearrange(x_reconstructed, "b (c h w) -> b c h w", c=c, h=h, w=w)
        return x_reconstructed
