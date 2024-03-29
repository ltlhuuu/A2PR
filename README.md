# Adaptive Advantage-guided Policy Regularization for Offline Reinforcement Learning

Code for ICML'24 submitted paper "Adaptive Advantage-guided Policy Regularization for Offline Reinforcement Learning".

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

For hopper:

```bash
python main.py  --seed 0 --env_id hopper-random-v2 --alpha 2.5 --vae_weight 1.5 --device cuda:0 --mask 1.5 --discount 0.995
python main.py  --seed 0 --env_id hopper-medium-v2 --alpha 2.5 --vae_weight 1.0 --device cuda:0 --mask 0.4 --discount 0.995 
```

## See result

```bash
tensorboard --logdir=result
```
