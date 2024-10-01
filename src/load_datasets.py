import torch
from torchvision import datasets, transforms


def load_dataset(dataset_class):
    def decorator(func):
        def wrapper(data_dir="./data", batch_size=64):
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),  # Normalize to range [-1, 1]
                ]
            )
            train_dataset = dataset_class(
                root=data_dir, train=True, transform=transform
            )
            test_dataset = dataset_class(
                root=data_dir, train=False, transform=transform
            )

            train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset, batch_size=batch_size, shuffle=True
            )
            test_loader = torch.utils.data.DataLoader(
                dataset=test_dataset, batch_size=batch_size, shuffle=False
            )
            return train_loader, test_loader

        return wrapper

    return decorator


@load_dataset(datasets.MNIST)
def get_mnist_loader(dataset, batch_size):
    pass


@load_dataset(datasets.FashionMNIST)
def get_fmnist_loader(dataset, batch_size):
    pass
