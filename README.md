# CP-Prompt: Domain-Incremental Continual Learning with Composition-Based Prompting

This repository is our PyTorch implementation of CP-Prompt.

## Requirements

```shell
pip install -r requirements.txt
```

## How to run

You can run the CP-Prompt with the following commands:

```shell
# CDDB
python main.py --config configs/prefix_one_prompt/cddb.json >> logs/prefix_one_prompt/cddb.log 2>&1
# CORe50
python main.py --config configs/prefix_one_prompt/core50.json >> logs/prefix_one_prompt/core50.log 2>&1
# Domainnet
python main.py --config configs/prefix_one_prompt/domainnet.json >> logs/prefix_one_prompt/domainnet.log 2>&1
```


## Acknowledgement

Our source code and data processing are built heavily based on the code of Roland ([https://github.com/snap-stanford/roland](https://github.com/iamwangyabin/S-Prompts)).

The data set download address is provided in the paper.
