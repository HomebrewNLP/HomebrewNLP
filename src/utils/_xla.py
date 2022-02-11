def load_torch_xla():
    # Note: install torch_xla and set model.xla.use_xla to "True"
    # to train TPU models.
    # See install instructions at: https://github.com/pytorch/xla
    #
    # Because XLA is optional and locks/unlocks threads, we'll import it once.

    global torch_xla, xm, xmp, pl
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader as pl