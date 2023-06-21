import os
from pydantic.dataclasses import dataclass

os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
import sys
import torch as t
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from einops import rearrange
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from tqdm.notebook import tqdm
from typing import List, Tuple, Dict, Callable
from PIL import Image
from IPython.display import display
from pathlib import Path
import torchinfo
import json
import pandas as pd
from jaxtyping import Float, Int
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part3_resnets"
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))
os.chdir(section_dir)

from part2_cnns.solutions import get_mnist, Linear, Conv2d, Flatten, ReLU, MaxPool2d
from part3_resnets.utils import print_param_count
import part3_resnets.tests as tests
from plotly_utils import line, plot_train_loss_and_test_accuracy_from_metrics

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


def main():
    pl.seed_everything(42)
    args = ConvNetTrainingArgs(max_epochs=100, weight_decay=1e-4)
    model = LitConvNet(args)
    logger = CSVLogger(save_dir=args.log_dir, name=args.log_name)

    trainer = pl.Trainer(max_epochs=args.max_epochs, logger=logger, log_every_n_steps=1)
    trainer.fit(model=model)

    # metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
    #
    # return line(
    #     metrics["train_loss"].values,
    #     x=metrics["step"].values,
    #     yaxis_range=[0, metrics["train_loss"].max() + 0.1],
    #     labels={"x": "Batches seen", "y": "Cross entropy loss"},
    #     title="ConvNet training on MNIST",
    #     width=800,
    #     hovermode="x unified",
    #     template="ggplot2",  # alternative aesthetic for your plots (-:
    # ).to_image(format="png")
    metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")

    return plot_train_loss_and_test_accuracy_from_metrics(
        metrics, "Training ConvNet on MNIST data"
    ).to_image(format="png")


if MAIN:
    main()
