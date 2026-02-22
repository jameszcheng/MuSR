# MuSR: Testing the Limits of Chain-of-thought with Multistep Soft Reasoning

### Creating murder mysteries that require multi-step reasoning with commonsense using ChatGPT!
By: Zayne Sprague, Xi Ye, Kaj Bostrom, Swarat Chaudhuri, and Greg Durrett.

zayne@utexas.edu {xi, kaj, swarat, gdurrett}@cs.utexas.edu

View the dataset on our [project website](https://zayne-sprague.github.io/MuSR/)!

Check out the [paper](https://arxiv.org/abs/2310.16049). Appearing at ICLR 2024 as a spotlight presentation!

<image src="./imgs/logo.png"></image>

## MuSR Eval

The datasets are in `datasets/{domain_name}.json`

### Install

1. Install `uv`: https://docs.astral.sh/uv/getting-started/installation/
2. Create and sync the environment:
   - `uv sync`
3. (Optional) run commands in the environment shell:
   - `uv run python -V`

### Evaluate


To run the evaluation script on the MuSR datasets:
```shell
export TOGETHER_API_KEY=key
uv run python -m eval.eval
```

You can edit the functionality of the evaluation in eval.py as well (including different prompting strategies, models, and more).

### [Optional] Install Redis for caching  

We cache all LLM calls (Together API and Hugging Face) with keys based on the prompt and model parameters to speed up evaluations.

To do this, we used [Redis](https://redis.io/docs/clients/python/)

Easiest way to install it is (for linux)
1. `apt-get install redis`
2. `redis-server`

Alternatively you can run our code without redis or disable the cache entirely by commenting out the lines `cache.enable()`.

### New models

Right now we support Together API chat/completion models and models published on Hugging Face.

Custom models made in PyTorch or Tensorflow will need an implementation that follows the `Model` class in `src/model/model.py`, similar to `src/model/hf.py` (for Hugging Face).

### New prompts and MuSR domain datasets

These are easily added to the `eval/eval.py` file.


## Overview of MuSR

<image src="./imgs/system_diagram.png"></image>


This repository holds the code for the paper _MuSR: Testing the Limits of Chain-of-thought with Multistep Soft Reasoning_

All MuSR datasets can be found in `{project_root}/datasets`. Follow the installation guide to get the datasets mentioned from the paper downloaded.

Major components for making the MuSR dataset can be found in `{project_root}/src`.  

Important classes and structures will have some example uses in their files (Madlib and LogicTree for example)

The scripts used to create a dataset can be found in `{project_root}/musr_dataset_scripts`

Evaluation scripts are in `{project_root}/eval`


## Generating a dataset

Every dataset creation script is in `{project_root}/musr_dataset_scripts`.  In those files are detailed instructions on how to create MuSR datasets as well as parameters for creating your own unique datasets!  Individual components that are used to create each dataset should be heavily documented as well.

To run a script:

```shell
export TOGETHER_API_KEY=key
uv run python musr_dataset_scripts/{dataset_script}.py
```
NOTE: We previously tested most of this with very strong frontier models. Quality may degrade if you switch to weaker models because prompts assume high-quality, well-formatted outputs.

This will produce a dataset file in `{project_root}/datasets` after it completes.

## Creating your own dataset

You can implement your own DatasetBuilder following the examples for the other domains.

For example, the important files used in creating murder mysteries are:

`{project_root}/src/dataset_builder.py`: The main file used for creating all datasets (shared functionality including the recursive reasoning tree expansion algorithm).

`{project_root}/src/dataset_types/murder_mystery_dataset.py`: Specific domain logic (and some prompts) for creating the murder mysteries.

`{project_root}/musr_dataset_scripts/create_murder_mysteries.py`: The main file that glues everything together (and includes some more prompts). 
