import numpy as np
import config

class StateDiscretizer:
    def __init__(self):
        self.d_bins = np.linspace(0, config.MAX_STATE_DISTANCE, config.NUM_DISTANCE_BINS)
        self.a_bins = np.linspace(-config.MAX_STATE_ANGLE, config.MAX_STATE_ANGLE, config.NUM_ANGLE_BINS)

    def discretize(self, state):
        d, a = state
        d_idx = np.digitize(d, self.d_bins)
        a_idx = np.digitize(a, self.a_bins)
        
        d_idx = min(config.NUM_DISTANCE_BINS - 1, max(0, d_idx - 1))
        a_idx = min(config.NUM_ANGLE_BINS - 1, max(0, a_idx - 1))
        return (d_idx, a_idx)