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

python main.py  --env_id hopper-medium-expert-v2 --alpha 2.5 --vae_weight 1.0 --device cuda:0 --mask 4.0 --discount 0.99 --seed 0

python main.py   --env_id hopper-medium-v2 --alpha 2.5 --vae_weight 1.0 --device cuda:0 --mask 0.4 --discount 0.99 --seed 0 not


python main.py   --env_id hopper-medium-replay-v2 --alpha 2.5 --vae_weight 1.5 --device cuda:0 --mask 0.5 --discount 0.99 --seed 0

python main.py   --env_id hopper-medium-v2 --alpha 2.5 --vae_weight 1.0 --device cuda:0 --mask 0.4 --discount 0.995 --seed 0 
```

## See result

```bash



python main.py   --env_id hopper-medium-v2 --alpha 2.5 --vae_weight 1.0 --device cuda:0 --mask 0.4 --discount 0.995 --seed 0 

python main.py   --env_id hopper-medium-replay-v2 --alpha 2.5 --vae_weight 1.5 --device cuda:0 --mask 0.5 --discount 0.99 --seed 0

python main.py  --env_id hopper-medium-expert-v2 --alpha 2.5 --vae_weight 1.0 --device cuda:0 --mask 4.0 --discount 0.99 --seed 0


python main.py   --env_id halfcheetah-medium-v2 --alpha 40.0 --vae_weight 1.0 --device cuda:0 --mask 1.0 --discount 0.99 --seed 0 

python main.py   --env_id halfcheetah-medium-replay-v2 --alpha 40.0 --vae_weight 1.5 --device cuda:0 --mask 0.8 --discount 0.995 --seed 0

python main.py   --env_id halfcheetah-medium-expert-v2 --alpha 40.0 --vae_weight 1.0 --device cuda:0 --mask 10.0 --discount 0.99 --seed 0


python main.py   --env_id walker2d-medium-v2 --alpha 2.5 --vae_weight 1.5 --device cuda:0 --mask 1.0 --discount 0.99 --seed 0

python main.py   --env_id walker2d-medium-replay-v2 --alpha 2.5 --vae_weight 1.5 --device cuda:0 --mask 1.2 --discount 0.995 --seed 0 

python main.py   --env_id walker2d-medium-expert-v2 --alpha 2.5 --vae_weight 1.0 --device cuda:0 --mask 0.8 --discount 0.99 --seed 0 

# current running

# python main.py   --env_id halfcheetah-medium-expert-v2 --alpha 40.0 --vae_weight 1.0 --device cuda:0 --mask 1.0 --discount 0.99 --seed 0
python main.py   --env_id walker2d-medium-replay-v2 --alpha 2.5 --vae_weight 1.5 --device cuda:0 --mask 1.5 --discount 0.995 --seed 0
python main.py   --env_id halfcheetah-medium-expert-v2 --alpha 40.0 --vae_weight 1.0 --device cuda:0 --mask 15.0 --discount 0.99 --seed 0
# python main.py   --env_id walker2d-medium-v2 --alpha 2.5 --vae_weight 1.0 --device cuda:0 --mask 1.5 --discount 0.99 --seed 0

# python main.py   --env_id walker2d-medium-v2 --alpha 2.5 --vae_weight 1.0 --device cuda:0 --mask 1.5 --discount 0.995 --seed 0 

# python main.py   --env_id walker2d-medium-replay-v2 --alpha 2.5 --vae_weight 1.5 --device cuda:0 --mask 1.2 --discount 0.99 --seed 0 


# wait for running

# python main.py   --env_id halfcheetah-medium-expert-v2 --alpha 40.0 --vae_weight 1.0 --device cuda:0 --mask 10.0 --discount 0.99 --seed 0
# python main.py   --env_id halfcheetah-medium-expert-v2 --alpha 40.0 --vae_weight 1.0 --device cuda:0 --mask 10.0 --discount 0.995 --seed 0

# python main.py   --env_id walker2d-medium-v2 --alpha 2.5 --vae_weight 1.0 --device cuda:0 --mask 1.5 --discount 0.99 --seed 0 
# python main.py   --env_id walker2d-medium-v2 --alpha 2.5 --vae_weight 1.0 --device cuda:0 --mask 1.5 --discount 0.995 --seed 0 





python main.py   --env_id walker2d-medium-expert-v2 --alpha 2.5 --vae_weight 1.0 --device cuda:0 --mask 0.8 --discount 0.99 --seed 0 
python main.py   --env_id walker2d-medium-expert-v2 --alpha 2.5 --vae_weight 1.0 --device cuda:0 --mask 0.8 --discount 0.995 --seed 0 

```

```bash
tensorboard --logdir=result
```
