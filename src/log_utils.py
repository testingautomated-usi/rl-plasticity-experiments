import numpy as np
from stable_baselines.results_plotter import (X_EPISODES, X_TIMESTEPS,
                                              X_WALLTIME)


def _ts2xy(timesteps, xaxis):
    """
    Decompose a timesteps variable to x ans ys

    :param timesteps: (Pandas DataFrame) the input data
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: (np.ndarray, np.ndarray) the x and y output
    """
    if xaxis == X_TIMESTEPS:
        x_var = np.cumsum(timesteps.l.values)
        y_var = timesteps.r.values
    elif xaxis == X_EPISODES:
        x_var = np.arange(len(timesteps))
        y_var = timesteps.l.values
    elif xaxis == X_WALLTIME:
        x_var = timesteps.t.values / 3600.0
        y_var = timesteps.r.values
    else:
        raise NotImplementedError
    return x_var, y_var
