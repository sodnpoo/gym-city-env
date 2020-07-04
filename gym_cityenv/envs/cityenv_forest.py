import numpy as np

from .cityenv import CityEnv

class CityEnvForest(CityEnv):
    """
    generates some randomy blobs of forest
    """

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
