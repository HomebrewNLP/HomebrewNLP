import sys
import typing

import torch
import yaml


class DataClass:
    pass


class Model(DataClass):
    features: int = 256
    depth: int = 64
    batch_size: int = 128
    sequence_length: int = 256
    activation_std: float = 0.5893595616022745  # std(relu(torch.randn((inf,)))) == 0.5893595616022745
    input_embedding_std: float = 1.
    position_embedding_std: float = 1.
    float16: bool = False
    device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    weight_shared_blocks: int = 1
    conv_kernel_size: int = 7
    feed_forward_intermediate_factor: float = 2.


class Dataset(DataClass):
    file_name: str = "out.tensor"
    classes: int = 256


class Optimizer(DataClass):
    learning_rate: float = 3e-4
    optimizer_type: str = "Adam"


class Train(DataClass):
    print_interval: int = 32
    features: int = 256
    gradient_accumulation_steps: int = 1


def init_class(instance: DataClass, config: typing.Dict[str, typing.Any]):
    for name, attr in instance.__dict__.items():
        if name not in config:
            continue
        if isinstance(attr, DataClass):
            init_class(attr, config[name])
            continue
        setattr(instance, name, config[name])


def serialize(instance: typing.Union[typing.Dict[str, DataClass], DataClass]):
    if isinstance(instance, DataClass):
        return serialize(instance.__dict__)
    return {k: serialize(v) if isinstance(v, DataClass) else v for k, v in instance.items()}


class Context(DataClass):
    def __init__(self, config: typing.Optional[typing.Dict[str, typing.Any]] = None):
        self.train = Train()
        self.optimizer = Optimizer()
        self.dataset = Dataset()
        self.model = Model()

        if len(sys.argv) > 1 and sys.argv[1].endswith('.yaml'):
            with open(sys.argv[1]) as f:
                cfg = f.read()
            init_class(self, yaml.safe_load(cfg))

        if config is not None:
            self.__dict__.update(config)
