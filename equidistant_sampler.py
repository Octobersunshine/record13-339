import pandas as pd
import numpy as np
import warnings
from typing import Optional, Union


def equidistant_sample(
    data: Union[pd.DataFrame, pd.Series],
    interval: int = 10,
    random_offset: bool = True,
    seed: Optional[int] = None,
    circular: bool = False,
    n_samples: Optional[int] = None,
) -> Union[pd.DataFrame, pd.Series]:
    if interval < 1:
        raise ValueError(f"interval 必须 >= 1，当前值: {interval}")
    if n_samples is not None and n_samples < 1:
        raise ValueError(f"n_samples 必须 >= 1，当前值: {n_samples}")

    n = len(data)
    if n == 0:
        return data.iloc[0:0]

    if interval > n:
        if circular:
            pass
        else:
            warnings.warn(
                f"采样间隔 interval={interval} 大于数据总量 n={n}，"
                f"将返回全部 {n} 条数据。建议将 interval 设置为 <= {n}。",
                UserWarning,
                stacklevel=2,
            )
            return data.reset_index(drop=True)

    if seed is not None:
        np.random.seed(seed)

    offset = np.random.randint(0, interval) if random_offset else 0

    if circular:
        if n_samples is None:
            n_samples = int(np.ceil(n / interval))
        raw_indices = offset + np.arange(n_samples) * interval
        indices = raw_indices % n
    else:
        indices = np.arange(offset, n, interval)
        if n_samples is not None:
            indices = indices[:n_samples]
        if len(indices) == 0:
            indices = np.array([offset % n])

    return data.iloc[indices].reset_index(drop=True)


class EquidistantSampler:
    def __init__(
        self,
        interval: int = 10,
        random_offset: bool = True,
        seed: Optional[int] = None,
        circular: bool = False,
        n_samples: Optional[int] = None,
    ):
        if interval < 1:
            raise ValueError(f"interval 必须 >= 1，当前值: {interval}")
        if n_samples is not None and n_samples < 1:
            raise ValueError(f"n_samples 必须 >= 1，当前值: {n_samples}")
        self.interval = interval
        self.random_offset = random_offset
        self.seed = seed
        self.circular = circular
        self.n_samples = n_samples
        self._last_offset: Optional[int] = None

    def sample(
        self,
        data: Union[pd.DataFrame, pd.Series],
        n_samples: Optional[int] = None,
    ) -> Union[pd.DataFrame, pd.Series]:
        n = len(data)
        if n == 0:
            return data.iloc[0:0]

        effective_n_samples = n_samples if n_samples is not None else self.n_samples

        if self.interval > n and not self.circular:
            warnings.warn(
                f"采样间隔 interval={self.interval} 大于数据总量 n={n}，"
                f"将返回全部 {n} 条数据。建议将 interval 设置为 <= {n}。",
                UserWarning,
                stacklevel=2,
            )
            self._last_offset = None
            return data.reset_index(drop=True)

        if self.seed is not None:
            np.random.seed(self.seed)

        offset = (
            np.random.randint(0, self.interval) if self.random_offset else 0
        )
        self._last_offset = offset

        if self.circular:
            if effective_n_samples is None:
                effective_n_samples = int(np.ceil(n / self.interval))
            raw_indices = offset + np.arange(effective_n_samples) * self.interval
            indices = raw_indices % n
        else:
            indices = np.arange(offset, n, self.interval)
            if effective_n_samples is not None:
                indices = indices[:effective_n_samples]
            if len(indices) == 0:
                indices = np.array([offset % n])

        return data.iloc[indices].reset_index(drop=True)

    @property
    def last_offset(self) -> Optional[int]:
        return self._last_offset
