import pathlib
import typing

import argh
import yaml

from src.dataclass import Context
from src.executable.inference import inference_cli
from src.executable.preprocess import preprocess_data
from src.executable.train import train_model
from src.utils.formatting import syntax_print
from src.utils.setup import setup_torch


def get_context(config_path: typing.Optional[str] = None) -> Context:
    '''
    Loads context from provided config. Otherwise loads default.
    '''
    if config_path is not None:
        config = pathlib.Path(config_path)
        assert config.suffix == '.yaml', 'Expected a .yaml file for config_path'
        ctx = Context(config_path=config)
    else:
        ctx = Context()
    return ctx


@argh.arg('-i', '--in_path', default='data.txt', help='Path for data to be preprocessed')
@argh.arg('-o', '--out_path', default='out.tensor', help='Path for data to be preprocessed')
def preprocess(in_path: str = 'data.txt', out_path: str = "out.tensor"):
    '''
    Processing original data into `out.tensor`
    '''
    preprocess_data(in_path, out_path)


@argh.arg('-c', '--config_path', default='configs/small.yaml', help='Path for the config file')
@argh.arg('-s', '--steps', default=0, help='Number of steps to take. 0 = infinite')
@argh.arg('-l', '--load_model', default=False, help='Whether to load an existing model checkpoint')
def train(config_path: typing.Optional[str] = None, steps: int = 0, load_model: bool = False):
    '''
    Trains a model given the config file.
    '''
    ctx = get_context(config_path)
    setup_torch(0, 'cuda' in ctx.model.device)

    dump = yaml.dump(ctx.serialize(), indent=4)
    syntax_print(dump, "yaml", title="Config")

    train_model(ctx, steps, load_model)


@argh.arg('-g', '--generated_tokens', default='20', help='Number of tokens to be generated after prompt')
@argh.arg('-t', '--temp', default='0.2', help='Temperature of the model.\nlower = consistency\nhigher = "creativity"')
@argh.arg('-c', '--config_path', help='Path for the config file')
def inference(generated_tokens: int = 20, temp: float = 0.2, config_path: str = None):
    '''
    Runs inference of input data on desired model
    '''
    assert config_path is not None, "Expected Config file!"

    ctx = get_context(config_path)

    inference_cli(ctx, float(temp), int(generated_tokens))


if __name__ == '__main__':
    parser = argh.ArghParser()
    parser.add_commands([preprocess, train, inference])
    parser.dispatch()
