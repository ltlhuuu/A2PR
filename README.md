# Adaptive Advantage-guided Policy Regularization for Offline Reinforcement Learning

Code for ICML'24 paper "Adaptive Advantage-guided Policy Regularization for Offline Reinforcement Learning".

If you find this repository useful for your research, please cite.

## Install dependency

```bash
pip install -r requirements.txt
```

Install the [D4RL](https://github.com/Farama-Foundation/D4RL) benchmark

```bash
git clone https://github.com/Farama-Foundation/D4RL.git
cd d4rl
pip install -e .
```

## Run experiment

```bash

python main.py   --env_id hopper-medium-v2 --alpha 2.5 --vae_weight 1.0 --device cuda:0 --mask 0.4 --discount 0.995 --seed 0 

python main.py   --env_id halfcheetah-medium-v2 --alpha 40.0 --vae_weight 1.0 --device cuda:0 --mask 1.0 --discount 0.99 --seed 0 

python main.py   --env_id walker2d-medium-v2 --alpha 2.5 --vae_weight 1.5 --device cuda:0 --mask 1.0 --discount 0.99 --seed 0

```

## See result

```bash
tensorboard --logdir=result
```
