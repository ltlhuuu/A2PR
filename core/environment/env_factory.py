import os

from core.environment.mountaincar import MountainCar
from core.environment.acrobot import Acrobot
from core.environment.lunarlander import LunarLander
from core.environment.halfcheetah import HalfCheetah
from core.environment.walker2d import Walker2d
from core.environment.hopper import Hopper
from core.environment.ant import Ant

class EnvFactory:
    @classmethod
    def create_env_fn(cls, cfg):
        if cfg.env_name == 'MountainCar':
            return lambda: MountainCar(cfg.seed)
        elif cfg.env_name == 'Acrobot':
            return lambda: Acrobot(cfg.seed)
        elif cfg.env_name == 'LunarLander':
            return lambda: LunarLander(cfg.seed)
        elif cfg.env_name == 'HalfCheetah':
            return lambda: HalfCheetah(cfg.seed)
        elif cfg.env_name == 'Walker2d':
            return lambda: Walker2d(cfg.seed)
        elif cfg.env_name == 'Hopper':
            return lambda: Hopper(cfg.seed)
        elif cfg.env_name == 'Ant':
            return lambda: Ant(cfg.seed)
        else:
            print(cfg.env_name)
            raise NotImplementedError