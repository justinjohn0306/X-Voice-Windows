# Training
## Prepare Dataset
Download corresponding dataset first, and fill in the path in scripts.

```bash
# Prepare the Emilia dataset
python src/train/datasets/prepare_emilia_speed.py
```


## Training & Finetuning

Once your datasets are prepared, you can start the training process.

### 1. Training script used for pretrained model

```bash
# setup accelerate config, e.g. use multi-gpu ddp, fp16
# will be to: ~/.cache/huggingface/accelerate/default_config.yaml     
accelerate config

# .yaml files are under src/f5_tts/configs directory
accelerate launch src/train/train.py --config-name SpeedPredict_Base.yaml

# possible to overwrite accelerate and hydra config
accelerate launch --mixed_precision=fp16 src/train/train.py --config-name SpeedPredict_Base.yaml ++datasets.batch_size_per_gpu=19200
```

### 2. W&B Logging

The `wandb/` dir will be created under path you run training/finetuning scripts.

By default, the training script does NOT use logging (assuming you didn't manually log in using `wandb login`).

To turn on wandb logging, you can either:

1. Manually login with `wandb login`: Learn more [here](https://docs.wandb.ai/ref/cli/wandb-login)
2. Automatically login programmatically by setting an environment variable: Get an API KEY at https://wandb.ai/authorize and set the environment variable as follows:

On Mac & Linux:

```
export WANDB_API_KEY=<YOUR WANDB API KEY>
```

On Windows:

```
set WANDB_API_KEY=<YOUR WANDB API KEY>
```
Moreover, if you couldn't access W&B and want to log metrics offline, you can set the environment variable as follows:

```
export WANDB_MODE=offline
```