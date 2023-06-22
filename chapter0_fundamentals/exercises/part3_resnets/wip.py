import os
import einops
import sys
import textwrap
from pathlib import Path
from typing import Tuple, Callable

from pydantic.dataclasses import dataclass
from torchvision import datasets, transforms

import pytorch_lightning as pl
import torch as t
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset


os.environ["ACCELERATE_DISABLE_RICH"] = "1"

# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part3_resnets"
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))
os.chdir(section_dir)

import part3_resnets.tests as tests
from part2_cnns.solutions import Linear, Conv2d, Flatten, ReLU, MaxPool2d

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(1, 32, kernel_size=3)
        self.relu1 = ReLU()
        self.mp1 = MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = Conv2d(32, 64, kernel_size=3)
        self.relu2 = ReLU()
        self.mp2 = MaxPool2d(kernel_size=2, stride=2)
        w = h = 7

        self.flatten = Flatten(start_dim=-3)

        n_h0 = 64 * w * h
        n_h1 = 128
        self.fc1 = Linear(in_features=n_h0, out_features=n_h1, bias=True)
        self.relu3 = ReLU()

        n_h2 = 10
        self.fc2 = Linear(in_features=n_h1, out_features=n_h2, bias=True)

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.mp1(self.relu1(self.conv1(x)))
        x = self.mp2(self.relu2(self.conv2(x)))
        x = self.fc2(self.relu3(self.fc1(self.flatten(x))))
        return x


data_augmentation_transform = transforms.Compose(
    [
        transforms.RandomRotation(degrees=15),
        transforms.RandomResizedCrop(size=28, scale=(0.8, 1.2)),
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

MNIST_TRANSFORM = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)


def get_mnist_augmented(
    subset: int = 1, train_transform: nn.Module = None, test_transform: nn.Module = None
):
    """Returns MNIST training data, sampled by the frequency given in `subset`."""
    mnist_trainset = datasets.MNIST(
        root="./data", train=True, download=True, transform=MNIST_TRANSFORM
    )
    mnist_testset = datasets.MNIST(
        root="./data", train=False, download=True, transform=MNIST_TRANSFORM
    )

    if subset > 1:
        mnist_trainset = Subset(
            mnist_trainset, indices=range(0, len(mnist_trainset), subset)
        )
        mnist_testset = Subset(
            mnist_testset, indices=range(0, len(mnist_testset), subset)
        )

    return mnist_trainset, mnist_testset


@dataclass(config=dict(arbitrary_types_allowed=True))
class ConvNetTrainingArgs:
    batch_size: int = 64
    max_epochs: int = 3
    optimizer: Callable[..., t.optim.Optimizer] = t.optim.Adam
    learning_rate: float = 1e-3
    weight_decay: float = 0
    log_dir: str = os.getcwd() + "/logs"
    log_name: str = "day3-convenet"
    log_every_n_steps: int = 1
    sample: int = 10


class LitConvNet(pl.LightningModule):
    def __init__(self, args: ConvNetTrainingArgs):
        super().__init__()
        self.convnet = ConvNet()
        self.args = args
        self.trainset, self.testset = get_mnist_augmented(
            subset=args.sample,
            train_transform=data_augmentation_transform,
        )

    def __step(
        self,
        batch: tuple[Tensor, Tensor],
        metric_name: str,
        metric: Callable[[t.Tensor, t.Tensor], float],
    ) -> t.Tensor:
        imgs, labels = batch
        logits = self.convnet(imgs)

        metric_value = metric(logits, labels)
        self.log(metric_name, metric_value)
        return metric_value

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> t.Tensor:
        return self.__step(batch, "train_loss", F.cross_entropy)

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        self.__step(
            batch,
            "accuracy",
            lambda logits, labels: (t.argmax(logits, -1) == labels)
            .float()
            .mean()
            .item(),
        )

    def configure_optimizers(self):
        return self.args.optimizer(
            self.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

    def val_dataloader(self):
        return DataLoader(self.testset, batch_size=self.args.batch_size, shuffle=False)

    def train_dataloader(self):
        return DataLoader(
            self.trainset, batch_size=self.args.batch_size, shuffle=True, num_workers=10
        )


class BatchNorm2d(nn.Module):
    # The type hints below aren't functional, they're just for documentation
    running_mean: Float[Tensor, "num_features"]
    running_var: Float[Tensor, "num_features"]
    num_batches_tracked: Int[Tensor, ""]  # This is how we denote a scalar tensor

    def __init__(self, num_features: int, eps=1e-05, momentum=0.1):
        """
        Like nn.BatchNorm2d with track_running_stats=True and affine=True.

        Name the learnable affine parameters `weight` and `bias` in that order.
        """
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.register_buffer("running_mean", t.zeros(num_features))
        self.register_buffer("running_var", t.ones(num_features))
        self.register_buffer("num_batches_tracked", t.tensor(0))

        self.weight = nn.Parameter(t.ones(num_features))
        self.bias = nn.Parameter(t.zeros(num_features))

    def __calc_running(self, old, observation):
        return old * (1 - self.momentum) + (observation * self.momentum)

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        Normalize each channel.

        Compute the variance using `torch.var(x, unbiased=False)`
        Hint: you may also find it helpful to use the argument `keepdim`.

        x: shape (batch, channels, height, width)
        Return: shape (batch, channels, height, width)
        """

        def make_broadcastable(x: t.Tensor):
            return einops.rearrange(x, "c -> 1 c 1 1")

        if self.training:
            self.num_batches_tracked += 1

            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            var = t.var(
                x,
                (0, 2, 3),
                keepdim=True,
                correction=0,
            )

            self.running_mean = self.__calc_running(self.running_mean, mean.squeeze())
            self.running_var = self.__calc_running(self.running_var, var.squeeze())
        else:
            mean = make_broadcastable(self.running_mean)
            var = make_broadcastable(self.running_var)

        weight = make_broadcastable(self.weight)
        bias = make_broadcastable(self.bias)

        normalized = (x - mean) / t.sqrt(var + self.eps)
        return normalized * weight + bias

    def extra_repr(self) -> str:
        return textwrap.dedent(
            f"""
            {self.num_features=}
            {self.eps=}
            {self.momentum=}
            """
        )


def main():
    tests.test_batchnorm2d_module(BatchNorm2d)
    tests.test_batchnorm2d_forward(BatchNorm2d)
    tests.test_batchnorm2d_running_mean(BatchNorm2d)
