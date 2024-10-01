## MNIST Autoencoder


This project demonstrates training an autoencoder on the MNIST dataset using PyTorch.


## Branches

- `main`: Contains the stable version of the code
- `development`: For developing new model architectures and testing features
- `reproducibility`: Ensures that paper results are reproducible

## Creating the Environment

```bash
conda env create -f environment.yml
conda activate mnist-autoencoder
```


## Exporting conda environment
`conda env export | grep -v "^prefix: " > environment.yml`

