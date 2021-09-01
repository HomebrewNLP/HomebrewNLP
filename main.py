import yaml

from src.dataclass import Context
from src.train import main
from src.utils import setup_torch

if __name__ == '__main__':
    # Get default context or from input yaml file 
    ctx = Context()
    # Configuring torch and setting seeds
    setup_torch(0)
    print(yaml.dump(ctx.serialize(), indent=4))
    # Training
    main(ctx)
