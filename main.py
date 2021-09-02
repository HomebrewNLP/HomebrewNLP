import argh
import pathlib
import typing
import yaml

from src.dataclass import Context
from src.utils.torch import setup_torch
from src.utils.formatting import syntax_print
from src.utils.preprocess import preprocess_data
from src.train import train_model


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
def train(config_path: typing.Optional[str] = None):
    '''
    Trains a model given the config file.
    '''
    ctx = get_context(config_path)
    setup_torch(0)

    dump = yaml.dump(ctx.serialize(), indent=4)
    syntax_print(dump, "yaml", title="Config")

    train_model(ctx)


@argh.arg('prompt', help='Input text to the model')
@argh.arg('-g', '--generated_tokens', default='20', help='Number of tokens to be generated after prompt')
@argh.arg('-t', '--temp', default='0.7', help='Temperature of the model.\nlower = consistency\nhigher = "creativity"')
@argh.arg('-c', '--config_path', help='Path for the config file')
def inference(prompt: str, generated_tokens: int = 20, temp: float = 0.7, config_path: str = None):
    '''
    Runs inference of input data on desired model
    '''
    assert config_path is not None, "Expected Config file!"

    ctx = get_context(config_path)

    # TODO: Load model (pretrained)
    # complete(ctx, model, prompt, temp, generated_tokens)

    raise NotImplementedError


if __name__ == '__main__':
    parser = argh.ArghParser()
    parser.add_commands([preprocess, train, inference])
    parser.dispatch()
