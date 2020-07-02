#!/usr/bin/env python

"""

"""

import os
import sys
import ctypes
from gi.repository import Gtk as gtk

# fix up path to include embedded micropolis
FILE_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(FILE_DIR, './micropolis/MicropolisCore/src')))

from pyMicropolis.gtkFrontend import main as frontend
from pyMicropolis.micropolisEngine import micropolisengine

import random

import gym
import numpy as np
from gym import spaces


class CityEnv(gym.Env):
    """
    """

    def __init__(self):
        self.MAP_X, self.MAP_Y = 16, 16

        self.engine, self.window = frontend.train()
        print(self.engine)
        print(self.window)

        self.window.playCity()
        self.engine.resume()
        self.engine.setGameMode('play')

        #self.init_funds = 2000000
        rand_funds = random.randint(5, 5)
        self.init_funds = rand_funds * 1000
        print(">>"*20, self.init_funds)
        self.engine.setFunds(self.init_funds)
        self.engine.setSpeed(3)
        self.engine.setPasses(5)
        #self.engine.setPasses(100)

        self.num_tools = 11

        self.num_obs_channels = 19 + 2

        '''
        ac_low = np.zeros((3))
        ac_high = np.array([self.num_tools - 1, self.MAP_X - 1, self.MAP_Y - 1])
        print(ac_low)
        print(ac_high)
        '''
        #self.action_space = spaces.Box(low=0, high=255, shape=(self.MAP_X, self.MAP_Y, self.num_tools), dtype=np.uint8)

        # each tool per map tile plus one for nop
        num_actions = 1 + self.MAP_X * self.MAP_Y * self.num_tools
        self.action_space = spaces.Discrete(num_actions)
        print(self.action_space)
        print(self.action_space.shape)
        print(len(self.action_space.shape))

        # TODO double check -1 .. 1 is okay
        low_obs = np.full((self.num_obs_channels, self.MAP_X, self.MAP_Y), fill_value=-1)
        high_obs = np.full((self.num_obs_channels, self.MAP_X, self.MAP_Y), fill_value=1)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=float)
        print(self.observation_space)
        print(self.observation_space.shape)
        print(len(self.observation_space.shape))

        self.engineTools = [ # tools in the order the engine sees them
            'Residential',
            'Commercial',
            'Industrial',
            'FireDept',
            'PoliceDept',
            'Query',
            'Wire',
            'Clear',
            'Rail',
            'Road',
            'Stadium',
            'Park',
            'Seaport',
            'CoalPowerPlant',
            'NuclearPowerPlant',
            'Airport',
            'Net',
            'Water',
            'Land',
            'Forest',
        ]
        self.tools = [ # tools how our agent sees them
            'Residential',
            'Commercial',
            'Industrial',
            'FireDept',
            'PoliceDept',
            # 'Query',
            'Clear',
            'Wire',
            # 'Land',
            'Rail',
            'Road',
            #'Stadium',
            #'Park',
            #'Seaport',
            'CoalPowerPlant',
            #'NuclearPowerPlant',
            #'Airport',
            #'Net',
            #'Water',
            'Land',
            #'Forest',
            #'Nil' # the agent takes no action
        ]


    def step(self, action):
        result = self.do(action)

        self.engine.tickEngine() # this calls simTick()

        reward = 0
        pop = sum([self.engine.resPop, self.engine.comPop, self.engine.indPop])
        done = pop == 0 and self.last_pop == 0 and self.engine.totalFunds <= 0
        if self.last_pop != pop:
            diff = pop - self.last_pop
            #print("X", self.last_pop, pop, diff)
            self.last_pop = pop
            reward = diff
            reward = max(0, reward)
        #print("reward:", reward)

        funds = self.engine.totalFunds / self.init_funds
        scalars = [
            funds,
            result,
        ]
        ob = self._get_state(scalars)

        self.steps += 1
        if self.steps > 1200:
            done = True

        #self.render()

        return ob, reward, done, {}

    def do(self, action):
        if action == 0:
            # no action
            return 0
        action -= 1
        x, y, z = np.unravel_index(action, (self.MAP_X, self.MAP_Y, self.num_tools))
        #print("action:", action, x, y, z)

        # map tool
        tool = self.tools[z]
        tool = self.engineTools.index(tool)
        #print(tool)

        # apply offset
        #x += self.MAP_XS
        #y += self.MAP_YS
        result = self.engine.toolDown(int(tool), int(x), int(y))
        return result


    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """

        self.steps = 0

        self.engine.setFunds(self.init_funds)

        self.last_pop = 0

        self.engine.autoBulldoze = False
        self.engine.autoBudget = True
        self.engine.roadPercent = 1.0
        self.engine.policePercent = 1.0
        self.engine.firePercent = 1.0
        self.engine.cityTax = 9

        self.new_map()

        self.engine.tickEngine()

        return self._get_state([1.0, 0])

    def render(self, mode="human", close=False):
        while gtk.events_pending():
            gtk.main_iteration()

    def seed(self, seed):
        random.seed(seed)
        np.random.seed

    def new_map(self):
        self.engine.generateMap()
        self.engine.clearMap()

        n = 10
        forest = 19
        for i in np.random.randint(16, size=(n, 2)):
            #print(i[0], i[1])
            x, y = int(i[0]), int(i[1])
            self.engine.toolDown(forest, x, y)
            self.engine.toolDown(forest, x+1, y+1)
            self.engine.toolDown(forest, x+1, y)
            self.engine.toolDown(forest, x, y+1)

            if np.random.rand() < 0.3:
                self.engine.toolDown(forest, x+0, y+2)
            if np.random.rand() < 0.3:
                self.engine.toolDown(forest, x+1, y+2)
            if np.random.rand() < 0.3:
                self.engine.toolDown(forest, x+2, y+0)
            if np.random.rand() < 0.3:
                self.engine.toolDown(forest, x+2, y+1)

    def _get_state(self, scalars):
        buffer = self.engine.getMapBuffer()
        p = ((ctypes.c_ushort * 100) * 120).from_address(int(buffer))
        a = np.ctypeslib.as_array(p).transpose()

        def mask(map, bitmask):
            return (np.bitwise_and(map, bitmask) > 0).astype(np.uint8)

        def mask_range(map, low, high):
            return ((map >= low) & (map <= high)).astype(np.uint8)

        state = None
        states = []
        playarea = a[:16,:16]
        with np.printoptions(threshold=np.inf, linewidth=100000):
            #print(">"*5, "raw")
            #print(playarea)

            #print(">"*5, "power")
            power = mask(playarea, micropolisengine.PWRBIT)
            #print(power)
            states.append(power)

            #print(">"*5, "conductive")
            conductive = mask(playarea, micropolisengine.CONDBIT)
            #print(conductive)
            states.append(conductive)

            # FIXME more thought into fire
            #print(">"*5, "burnable")
            #burnable = mask(playarea, micropolisengine.BURNBIT)
            #print(burnable)
            #states.append(burnable)

            #print(">"*5, "bulldozable")
            #bulldozable = mask(playarea, micropolisengine.BULLBIT)
            #print(bulldozable)
            #states.append(bulldozable)

            #print(">"*5, "zonecentres")
            zonecentres = mask(playarea, micropolisengine.ZONEBIT)
            #print(zonecentres)
            states.append(zonecentres)

            #print(">"*5, "lomask")
            lomask = np.bitwise_and(playarea, micropolisengine.LOMASK)
            #print(lomask)

            #print(">"*5, "residential")
            residential = mask_range(lomask, micropolisengine.RESBASE, micropolisengine.COMBASE-1)
            #print(residential)
            states.append(residential)

            #print(">"*5, "residential density")
            res_values = np.ma.masked_array(lomask.astype(np.float16), ~residential.astype(np.bool))
            res_density = ((res_values - micropolisengine.RESBASE) // 9) # 0 to 15
            res_density = res_density * (1/15) # scale to 0 to 1
            res_density = res_density.filled(0)
            #print(res_density)
            states.append(res_density)

            #print(">"*5, "commercial")
            commercial = mask_range(lomask, micropolisengine.COMBASE, micropolisengine.COMLAST)
            #print(commercial)
            states.append(commercial)

            #print(">"*5, "commercial density")
            com_values = np.ma.masked_array(lomask.astype(np.float16), ~commercial.astype(np.bool))
            com_density = ((com_values - micropolisengine.COMBASE) // 9) # 0 to 15
            com_density = com_density * (1/15) # scale to 0 to 1
            com_density = com_density.filled(0)
            #print(com_density)
            states.append(com_density)

            #print(">"*5, "industrial")
            industrial = mask_range(lomask, micropolisengine.INDBASE, micropolisengine.PORTBASE-1)
            industrial2 = mask_range(lomask, micropolisengine.INDBASE2, micropolisengine.COALSMOKE1-1)
            industrial = industrial | industrial2
            #print(industrial)
            states.append(industrial)

            #print(">"*5, "industrial density")
            ind_values = np.ma.masked_array(lomask.astype(np.float16), ~industrial.astype(np.bool))
            ind_density = ((ind_values - micropolisengine.INDBASE) // 9) # 0 to 15
            ind_density = ind_density * (1/15) # scale to 0 to 1
            ind_density = ind_density.filled(0)
            #print(ind_density)
            states.append(ind_density)

            #print(">"*5, "burning")
            burning = mask_range(lomask, micropolisengine.FIREBASE, micropolisengine.LASTFIRE)
            #print(burning)
            states.append(burning)

            #print(">"*5, "road")
            road = mask_range(lomask, micropolisengine.ROADBASE, micropolisengine.LASTROAD)
            #print(road)
            states.append(road)

            #print(">"*5, "rail") #rail wires are in the POWER range
            rail = mask_range(lomask, micropolisengine.RAILBASE, micropolisengine.LASTRAIL)
            rail_wires = ((lomask == micropolisengine.RAILHPOWERV) | (lomask == micropolisengine.RAILVPOWERH))
            rail = rail | rail_wires
            #print(rail)
            states.append(rail)

            #print(">"*5, "wires") #road wires are in the ROAD range
            wires = mask_range(lomask, micropolisengine.POWERBASE, micropolisengine.LASTPOWER)
            road_wires = ((lomask == micropolisengine.HROADPOWER) | (lomask == micropolisengine.VROADPOWER))
            wires = wires | road_wires
            #print(wires)
            states.append(wires)

            #print(">"*5, "powerplant")
            powerplant = mask_range(lomask, micropolisengine.COALBASE, micropolisengine.LASTPOWERPLANT) # coal
            smoke = mask_range(lomask, micropolisengine.COALSMOKE1, micropolisengine.FOOTBALLGAME1-1)
            powerplant = powerplant | smoke
            #print(powerplant)
            states.append(powerplant)

            #print(">"*5, "firestation")
            firestation = mask_range(lomask, micropolisengine.FIRESTBASE, micropolisengine.POLICESTBASE-1)
            #print(firestation)
            states.append(firestation)

            #print(">"*5, "policestation")
            policestation = mask_range(lomask, micropolisengine.POLICESTBASE, micropolisengine.STADIUMBASE-1)
            #print(policestation)
            states.append(policestation)

            #print(">"*5, "rubble")
            rubble = mask_range(lomask, micropolisengine.RUBBLE, micropolisengine.LASTRUBBLE)
            #print(rubble)
            states.append(rubble)

            #print(">"*5, "trees")
            trees = mask_range(lomask, micropolisengine.TREEBASE, micropolisengine.WOODS_HIGH)
            #print(trees)
            states.append(trees)

            #print(">"*5, "pollution")
            pollution = np.zeros_like(playarea.astype(np.float16))
            # coal power plant
            pollution[((lomask >= micropolisengine.PORTBASE) & (lomask <= micropolisengine.LASTPOWERPLANT))] = 1.0
            pollution[((lomask >= micropolisengine.COALSMOKE1) & (lomask < micropolisengine.FOOTBALLGAME1))] = 1.0
            # industrial
            pollution[((lomask > micropolisengine.LASTIND) & (lomask < micropolisengine.PORTBASE))] = 0.5
            pollution[((lomask > micropolisengine.INDBASE2) & (lomask < micropolisengine.COALSMOKE1))] = 0.5
            # low traffic
            pollution[((lomask >= micropolisengine.LTRFBASE) & (lomask < micropolisengine.HTRFBASE))] = 0.25
            # high traffic
            pollution[((lomask >= micropolisengine.HTRFBASE) & (lomask < micropolisengine.POWERBASE))] = 0.75
            # fire
            pollution[((lomask >= micropolisengine.FIREBASE) & (lomask <= micropolisengine.LASTFIRE))] = 0.9
            #print(pollution)
            states.append(pollution)

            expanded_states = []
            for s in states:
                expanded = np.expand_dims(s, axis=0)
                expanded_states.append(expanded)

        for s in scalars:
            expanded = np.full_like(expanded_states[0].astype(np.float16), s)
            expanded_states.append(expanded)

        state = np.concatenate(expanded_states, axis=0)
        state = state.astype(np.float16)

        return state
