from src.dataclass import Context
from src.train import main
import yaml

if __name__ == '__main__':
    ctx = Context()
    print(yaml.dump(ctx.serialize(), indent=4))
    main(ctx)
