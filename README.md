# Adaptive Advantage-guided Policy Regularization for Offline Reinforcement Learning

This repo is the official implementation of ICML'24 paper "Adaptive Advantage-guided Policy Regularization for Offline Reinforcement Learning".

If you find this repository useful for your research, please cite:

```bib
@inproceedings{
    A2PR,
    title={Adaptive Advantage-guided Policy Regularization for Offline Reinforcement Learning},
    author={Tenglong Liu and Yang Li and Yixing Lan and Hao Gao and Wei Pan and Xin Xu},
    booktitle={International Conference on Machine Learning},
    year={2024}
}
```

## Install dependency
Environment configuration and dependencies are available in `environment.yaml` and `requirements.txt`.

First, create the conda environment.
```bash
conda env create -f environment.yaml
conda activate A2PR
```

Then install the remaining requirements (with MuJoCo already downloaded, if not see [here](#MuJoCo-installation)): 
```bash
pip install -r requirements.txt
```


Install the [D4RL](https://github.com/Farama-Foundation/D4RL) benchmark

```bash
git clone https://github.com/Farama-Foundation/D4RL.git
cd d4rl
pip install -e .
```
### MuJoCo installation
Download MuJoCo:
```bash
mkdir ~/.mujoco
cd ~/.mujoco
wget https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
tar -zxvf mujoco210-linux-x86_64.tar.gz
cd mujoco210
wget https://www.roboti.us/file/mjkey.txt
```
Then add the following line to `.bashrc`:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
```

## Running experiments
In the following, you can use the illustrative examples to run the experiments.

```bash
python main.py   --env_id hopper-medium-v2 --alpha 2.5 --vae_weight 1.0 --device cuda:0 --mask 0.4 --discount 0.995 --seed 0 

python main.py   --env_id halfcheetah-medium-v2 --alpha 40.0 --vae_weight 1.0 --device cuda:0 --mask 1.0 --discount 0.99 --seed 0 

python main.py   --env_id walker2d-medium-v2 --alpha 2.5 --vae_weight 1.5 --device cuda:0 --mask 1.0 --discount 0.99 --seed 0
```

## See result

```bash
tensorboard --logdir='Your output path'
```
### For example
```bash
tensorboard --logdir=result
```
