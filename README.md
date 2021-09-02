# HomebrewNLP

## Installation

You'll need CUDA, CUDA-Toolkit and NCCL to match the versions used in pytorch and deepspeed. At the time of
writing, `torch==1.9.0+cu111` + CUDA11.1 is one of the most popular combinations. CUDA11.2 will not work. You also need
to install the requirements with `USE_NCCL=0` as FastMoE will otherwise attempt to compile tools for distributed work
which sometimes fail to compile.

## Overview

> "...Our goal is to open up the space by combining every form of efficient training we have. If we throw enough tradeoffs against it, a model of this size (GPT-3) should be trainable on commodity hardware (<1k if purchased as upgrades) ... Compute-memory tradeoffs (like MOE) aren't enough ... we want more efficient training using extragradient methods and better optimizers (Shampoo)" - **Lucas Nestler**

## Example Command

```BASH
python3 main.py configs/small.yaml
```

---

[![DeepSource](https://deepsource.io/gh/HomebrewNLP/HomebrewNLP.svg/?label=active+issues&show_trend=true&token=sAQ42SRyNPilkjj82sQd88ea)](https://deepsource.io/gh/HomebrewNLP/HomebrewNLP/?ref=repository-badge)
| [Dataset](https://drive.google.com/file/u/1/d/1aoW3KI2E3nK7B28RE6I6_oDtNidTvoc2/view?usp=sharing)
| [Discord](https://discord.gg/JSGG6Abcyx)

