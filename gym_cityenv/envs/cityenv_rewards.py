import numpy as np
import ctypes

# fix up path to include embedded micropolis
import os
import sys
FILE_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(FILE_DIR, './micropolis/MicropolisCore/src')))

from pyMicropolis.micropolisEngine import micropolisengine

from .cityenv import CityEnv

class CityEnvRewardByPower(CityEnv):
    """
    modulates the reward by the percentage of structures powered
    powered vs conductivfrom pyMicropolis.gtkFrontend import main as frontend
e
    """

    def calc_power_pct(self):
        buffer = self.engine.getMapBuffer()
        p = ((ctypes.c_ushort * 100) * 120).from_address(int(buffer))
        a = np.ctypeslib.as_array(p).transpose()

        def mask(map, bitmask):
            return (np.bitwise_and(map, bitmask) > 0).astype(np.uint8)

        playarea = a[:16,:16]

        #print(">"*5, "power")
        power = mask(playarea, micropolisengine.PWRBIT)
        #print(power)

        #print(">"*5, "conductive")
        conductive = mask(playarea, micropolisengine.CONDBIT)
        #print(conductive)

        conductive_count = np.count_nonzero(conductive)
        power_count = np.count_nonzero(power)
        #print("*"*20, power_count, conductive_count)
        if conductive_count > 0:
            pc = float(power_count)/float(conductive_count)
        else:
            pc = 0.0

        #print("*"*20, power_count, conductive_count, pc)
        return pc

    def reward(self):
        reward = super().reward()
        power_pc = self.calc_power_pct()
        reward *= power_pc
        return reward
