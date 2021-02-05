from tqdm.auto import tqdm

from custom_callbacks import ProgressBarCallback
from custom_callbacks3 import ProgressBarCallback as ProgressBarCallback3


class ProgressBarManager(object):
    def __init__(self, sb_version: str, total_timesteps):  # init object with total timesteps
        self.pbar = None
        self.sb_version = sb_version
        self.total_timesteps = total_timesteps

    def __enter__(self):  # create the progress bar and callback, return the callback
        self.pbar = tqdm(total=self.total_timesteps)

        if self.sb_version == "sb3":
            return ProgressBarCallback3(self.pbar)
        return ProgressBarCallback(self.pbar)

    def __exit__(self, exc_type, exc_val, exc_tb):  # close the callback
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()
