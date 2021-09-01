import argh
from pathlib import Path
import typing
import yaml

from src.dataclass import Context
from src.train import main
from src.utils import setup_torch
from src.formatting import syntax_print


def get_context(config_path: typing.Optional[str] = None) -> Context:
    '''
    Loads context from provided config. Otherwise loads default.
    '''
    if config_path is not None:
        config = Path(config_path)
        assert config.suffix == '.yaml', 'Expected a .yaml file for config_path'
        ctx = Context(config_path=config)
    else:
        ctx = Context()
    return ctx


def preprocess():
    '''
    Processing original data into `out.tensor`
    '''
    # TODO: Add to CLI.
    raise NotImplementedError


@argh.arg('-c', '--config_path', default='configs/small.yaml', help='Path for the config file')
def train(config_path: typing.Optional[str] = None):
    '''
    Trains a model given the config file.
    '''
    ctx = get_context(config_path)
    setup_torch(0)

    dump = yaml.dump(ctx.serialize(), indent=4)
    syntax_print(dump, "yaml", title="Config")

    main(ctx)


def inference():
    # def inference(ctx: Context, model: torch.nn.Module, prompt: str, temperature: float, generated_tokens: int) -> str:
    '''
    Runs inference of input data on desired model
    '''
    # TODO:
    raise NotImplementedError


if __name__ == '__main__':
    parser = argh.ArghParser()
    parser.add_commands([preprocess, train, inference])
    parser.dispatch()
