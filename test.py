import gc
import time

import memory_profiler
import torch

import module


def test(batch_size=1):
    sequence_length = [64, 256, 1024, 2048]
    features = [64]
    delays = [2]

    def _max_len(seq):
        return max(map(len, map(str, seq)))

    seq_len = _max_len(sequence_length)
    fet_len = _max_len(features)
    del_len = _max_len(delays)

    def _full(seq, feat, delay):
        mod = module.RevRNN(256, feat, delay=delay)
        start_time = time.time()
        inp = torch.randn((batch_size, seq, 256), requires_grad=True)
        out = mod(inp)
        out.mean().backward()
        del mod
        del inp
        del out
        gc.collect()
        return time.time() - start_time, batch_size * seq * 256

    for seq in sequence_length:
        for feat in features:
            for delay in delays:
                profile = memory_profiler.LineProfiler(backend='psutil')
                exec_time, num_elem = profile(func=_full)(seq, feat, delay)
                data_memory = num_elem * 4 * 2 ** -20  # 4 bytes per elem, everything in MiB
                max_memory = max(max(y for _, (y, _) in x) for _, x in profile.code_map.items())
                print(f"Batch: {batch_size} - "
                      f"Seq: {seq:{seq_len}d} - Feat: {feat:{fet_len}d} - Delay: {delay:{del_len}d} - "
                      f"Total Memory: {max_memory:8.2f} MiB - Data Memory: {data_memory:8.2f} MiB - "
                      f"Normalized Memory: {max_memory - data_memory:8.2f} MiB - "
                      f"Total Time: {exec_time:7.2f}s")


if __name__ == "__main__":
    test(4096)
