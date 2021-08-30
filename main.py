import yaml

from src.dataclass import Context
from src.train import main
from src.utils import setup_torch

if __name__ == '__main__':
    ctx = Context()
    setup_torch(0)
    print(yaml.dump(ctx.serialize(), indent=4))
    main(ctx)
