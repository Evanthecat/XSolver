import numpy as np

class Particles:
    def __init__(self, rest_pos, rest_imp, rest_times, pos=None, jac=np.identity(3)):
        self.rest_positions = rest_pos
        self.rest_impulses = rest_imp
        self.rest_times = rest_times

        if type(pos) == np.array:
            self.positions = pos
        else:
            self.positions = rest_pos
        self.jacobians = jac