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
        return serialize({key: value for key, value in attributes.items() if not isinstance(value, typing.Callable)})
    return {k: serialize(v) if isinstance(v, DataClass) else v for k, v in instance.items()}


class MoE(DataClass):
    num_experts: int = 16
    use_in_input: bool = False
    use_in_output: bool = False


class Model(DataClass):
    features: int = 256
    momentumnet_beta: float = 0.99  # The higher this is, the more numerically stable. BUT also lower impact per layer
    depth: int = 64
    batch_size: int = 128
    sequence_length: int = 256
    activation_std: float = 0.5893595616022745  # std(relu(torch.randn((inf,)))) == 0.5893595616022745
    input_embedding_std: float = 1.
    position_embedding_std: float = 1.
    float16: bool = False
    device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    conv_kernel_size: int = 7
    feed_forward_intermediate_factor: float = 2.
    dropout_probability: float = 0.
    bottleneck_group = 1  # not all group counts are possible. it has to be divide self.features without residual
    moe: MoE = MoE()


class Log(DataClass):
    deepspeed_steps_per_print: int = 2 ** 20
    wall_clock_breakdown: bool = False
    dump_state: bool = False
    loss_steps_per_print: int = 32


class Dataset(DataClass):
    file_name: str = "out.tensor"
    classes: int = 256
    shuffle: bool = False
    num_workers: int = 4
    pin_memory: bool = False
    prefetch_factor: int = 2


class Offload(DataClass):
    device: str = "cpu"
    pin_memory: bool = True


class Zero(DataClass):
    cpu_offload: bool = True
    contiguous_gradients: bool = False
    overlap_comm: bool = True
    offload_param: Offload = Offload()
    offload_optimizer: Offload = Offload()
    stage3_max_live_parameters: float = 1
    stage3_max_reuse_distance: float = 1
    stage3_prefetch_bucket_size: float = 1
    stage3_param_persistence_threshold: float = 1


class OneCycle(DataClass):
    cycle_min_lr: float = 3e-4  # Base learning rate used at the start and end of cycle.
    cycle_max_lr: float = 1e-3  # Learning rate used in the middle of the cycle. Can be smaller than cycle_min_lr
    decay_lr_rate: float = 1e-4  # Decay rate for learning rate.
    cycle_first_step_size: int = 2048  # Number of training iterations in the increasing half of a cycle.
    cycle_second_step_size: typing.Optional[int] = None  # steps in second phase. None -> cycle_first_step_size
    cycle_first_stair_count: int = 0  # Number of stairs in first phase. 0 means staircase disabled
    cycle_second_stair_count: typing.Optional[int] = None  # Number of stairs in second phase
    decay_step_size: int = 2  # Every how many steps to decay lr. 0 -> no decay
    cycle_momentum: bool = True  # Whether to cycle `momentum` inversely to learning rate.
    cycle_min_mom: float = 0.8  # Initial momentum which is the lower boundary in the cycle for each parameter group.
    cycle_max_mom: float = 0.9  # Upper momentum boundaries in the cycle for each parameter group.
    decay_mom_rate: float = 0  # Decay rate for momentum
    last_batch_iteration: int = -1  # The index of the last batch. This parameter is used when resuming a training job.


class Optimizer(DataClass):
    type: str = "AdamW"
    gradient_accumulation_steps: int = 1
    one_cycle: OneCycle = OneCycle()
    beta2: float = 0.9999  # beta1 is controlled by one_cycle
    epsilon: float = 1e-8
    weight_decay: float = 0.01
    gradient_clipping: float = 1.
    zero: Zero = Zero()


class Eval(DataClass):
    cache: bool = True


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
        self.eval = Eval()

        if len(sys.argv) > 1 and sys.argv[1].endswith('.yaml'):
            with open(sys.argv[1]) as f:
                cfg = f.read()
            init_class(self, yaml.safe_load(cfg))

        if config is not None:
            self.__dict__.update(config)
