import pandas as pd
import numpy as np
from typing import Optional, Union


def equidistant_sample(
    data: Union[pd.DataFrame, pd.Series],
    interval: int = 10,
    random_offset: bool = True,
    seed: Optional[int] = None,
) -> Union[pd.DataFrame, pd.Series]:
    if interval < 1:
        raise ValueError(f"interval 必须 >= 1，当前值: {interval}")

    n = len(data)
    if n == 0:
        return data.iloc[0:0]

    if seed is not None:
        np.random.seed(seed)

    offset = np.random.randint(0, interval) if random_offset else 0

    indices = np.arange(offset, n, interval)
    if len(indices) == 0:
        indices = np.array([offset % n])

    return data.iloc[indices].reset_index(drop=True)


class EquidistantSampler:
    def __init__(
        self,
        interval: int = 10,
        random_offset: bool = True,
        seed: Optional[int] = None,
    ):
        if interval < 1:
            raise ValueError(f"interval 必须 >= 1，当前值: {interval}")
        self.interval = interval
        self.random_offset = random_offset
        self.seed = seed
        self._last_offset: Optional[int] = None

    def sample(
        self, data: Union[pd.DataFrame, pd.Series]
    ) -> Union[pd.DataFrame, pd.Series]:
        if self.seed is not None:
            np.random.seed(self.seed)

        n = len(data)
        if n == 0:
            return data.iloc[0:0]

        offset = (
            np.random.randint(0, self.interval) if self.random_offset else 0
        )
        self._last_offset = offset

        indices = np.arange(offset, n, self.interval)
        if len(indices) == 0:
            indices = np.array([offset % n])

        return data.iloc[indices].reset_index(drop=True)

    @property
    def last_offset(self) -> Optional[int]:
        return self._last_offset
