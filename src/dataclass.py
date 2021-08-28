import sys
import typing

import torch
import yaml


class DataClass:
    def serialize(self):
        return serialize(self)


def serialize(instance: typing.Union[DataClass, typing.Dict[str, typing.Any]]):
    if isinstance(instance, DataClass):
        attributes = {key: getattr(instance, key) for key in dir(instance)
                      if not key.startswith('_') and not key.endswith('_')}
        return serialize({key: value for key, value in attributes.items() if not isinstance(key, typing.Callable)})
    return {k: serialize(v) if isinstance(v, DataClass) else v for k, v in instance.items()}


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


class Log(DataClass):
    deepspeed_steps_per_print: int = 2 ** 20
    wall_clock_breakdown: bool = False
    dump_state: bool = False
    loss_steps_per_print: int = 32


class Dataset(DataClass):
    file_name: str = "out.tensor"
    classes: int = 256


class Optimizer(DataClass):
    learning_rate: float = 3e-4
    optimizer_type: str = "Adam"
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 2048


def init_class(instance: DataClass, config: typing.Dict[str, typing.Any]):
    for name in dir(instance):
        if name.startswith("_") or name.endswith("_") or name not in config:
            continue
        attr = getattr(instance, name)
        if isinstance(attr, DataClass):
            init_class(attr, config[name])
            continue
        setattr(instance, name, config[name])


class Context(DataClass):
    def __init__(self, config: typing.Optional[typing.Dict[str, typing.Any]] = None):
        self.log = Log()
        self.optimizer = Optimizer()
        self.dataset = Dataset()
        self.model = Model()
        if len(sys.argv) > 1 and sys.argv[1].endswith('.yaml'):
            with open(sys.argv[1]) as f:
                cfg = f.read()
            init_class(self, yaml.safe_load(cfg))

        if config is not None:
            self.__dict__.update(config)
