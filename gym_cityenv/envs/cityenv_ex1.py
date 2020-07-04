import numpy as np
import ctypes

from .cityenv_forest import CityEnvForest
from .cityenv_rewards import CityEnvRewardByPower


class CityEnvEx1(CityEnvForest, CityEnvRewardByPower):
    """
    modulates the reward by the percentage of structures powered
    powered vs conductivfrom pyMicropolis.gtkFrontend import main as frontend
    """
