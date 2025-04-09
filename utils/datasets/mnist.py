import ssl
import torchvision
from icecream import ic

ssl._create_default_https_context = ssl._create_unverified_context
if __name__ == "__main__":
    dataset = torchvision.datasets.MNIST(root="data/", download=True)
    ic("Dataset loaded")
