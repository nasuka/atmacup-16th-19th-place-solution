import os
import random
import time
from contextlib import contextmanager

import numpy as np


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print("[{}] done in {} s".format(name, time.time() - t0))


def set_seed(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
