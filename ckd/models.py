# Author: Ziwei Wang

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelConfig:
    channels: int = 22
    samples: int = 1000
    sampling_rate: int = 250
    f1: int = 4
    d: int = 2
    f2: int = 8
    num_classes: int = 2

    def __post_init__(self):
        for name in ("channels", "samples", "sampling_rate", "f1", "d", "f2", "num_classes"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.samples < 32:
            raise ValueError("samples must be at least 32")
        if self.sampling_rate < 4:
            raise ValueError("sampling_rate must be at least 4")
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least 2")


class EEGNet_block1(nn.Module):
    def __init__(self, n_classes, Chans, Samples, kernLenght, F1, D, F2, dropoutRate, norm_rate):
        super(EEGNet_block1, self).__init__()
        self.n_classes = n_classes
        self.Chans = Chans
        self.Samples = Samples
        self.kernLenght = kernLenght
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.dropoutRate = dropoutRate
        self.norm_rate = norm_rate
        self.block1 = nn.Sequential(
            nn.ZeroPad2d((self.kernLenght // 2 - 1, self.kernLenght - self.kernLenght // 2, 0, 0)),
            nn.Conv2d(in_channels=1, out_channels=self.F1, kernel_size=(1, self.kernLenght), stride=1, bias=False),
            nn.BatchNorm2d(num_features=self.F1),
            nn.Conv2d(in_channels=self.F1, out_channels=self.F1 * self.D, kernel_size=(self.Chans, 1), groups=self.F1, bias=False),
            nn.BatchNorm2d(num_features=self.F1 * self.D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(p=self.dropoutRate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.block1(x)
        return output


class EEGNet_block2(nn.Module):
    def __init__(self, n_classes, Chans, Samples, kernLenght, F1, D, F2, dropoutRate, norm_rate):
        super(EEGNet_block2, self).__init__()
        self.n_classes = n_classes
        self.Chans = Chans
        self.Samples = Samples
        self.kernLenght = kernLenght
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.dropoutRate = dropoutRate
        self.norm_rate = norm_rate
        self.block2 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(in_channels=self.F1 * self.D, out_channels=self.F1 * self.D, kernel_size=(1, 16), stride=1, groups=self.F1 * self.D, bias=False),
            nn.Conv2d(in_channels=self.F1 * self.D, out_channels=self.F2, kernel_size=(1, 1), stride=1, bias=False),
            nn.BatchNorm2d(num_features=self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.dropoutRate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.block2(x)
        output = output.reshape(output.size(0), -1)
        return output


class FC_xy(nn.Module):
    def __init__(self, nn_in, nn_out):
        super(FC_xy, self).__init__()
        self.nn_out = nn_out
        self.fc = nn.Linear(nn_in, nn_out)

    def forward(self, x):
        y = self.fc(x)
        return x, y


def _module_kwargs(config: ModelConfig):
    return dict(
        n_classes=config.num_classes,
        Chans=config.channels,
        Samples=config.samples,
        kernLenght=int(config.sampling_rate // 2),
        F1=config.f1,
        D=config.d,
        F2=config.f2,
        dropoutRate=0.25,
        norm_rate=0.5,
    )


def build_modules(config: ModelConfig):
    kwargs = _module_kwargs(config)
    embedding = EEGNet_block1(**kwargs)
    kwargs["Chans"] = -1
    encoder = EEGNet_block2(**kwargs)
    classifier = FC_xy(config.f2 * (config.samples // 4 // 8), config.num_classes)
    return embedding, encoder, classifier


def build_multisource_modules(configs: dict[str, ModelConfig]):
    if not configs:
        raise ValueError("configs must contain at least one source")
    first = next(iter(configs.values()))
    for config in configs.values():
        for name in ("samples", "sampling_rate", "f1", "d", "f2", "num_classes"):
            if getattr(config, name) != getattr(first, name):
                raise ValueError(f"all source configurations must have the same {name}")
    embeddings = nn.ModuleDict({name: EEGNet_block1(**_module_kwargs(config)) for name, config in configs.items()})
    kwargs = _module_kwargs(first)
    kwargs["Chans"] = -1
    encoder = EEGNet_block2(**kwargs)
    classifier = FC_xy(first.f2 * (first.samples // 4 // 8), first.num_classes)
    return embeddings, encoder, classifier


class CKDModel(nn.Module):
    def __init__(self, embedding: nn.Module, encoder: nn.Module, classifier: nn.Module):
        super().__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x: torch.Tensor):
        return self.classifier(self.encoder(self.embedding(x)))
