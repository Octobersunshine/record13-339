import pandas as pd
import numpy as np
from equidistant_sampler import equidistant_sample, EquidistantSampler


def test_basic_sampling():
    df = pd.DataFrame({"value": range(100)})
    result = equidistant_sample(df, interval=10, random_offset=False, seed=42)
    expected = df.iloc[::10].reset_index(drop=True)
    assert result.equals(expected), f"基本采样失败: {result.tolist()}"
    print(f"[PASS] 基本采样(无偏移): 100条 → {len(result)}条, 索引={result['value'].tolist()}")


def test_random_offset():
    df = pd.DataFrame({"value": range(100)})
    result = equidistant_sample(df, interval=10, random_offset=True, seed=42)
    offset = result["value"].iloc[0]
    assert offset < 10, f"偏移应 < interval, 实际: {offset}"
    diffs = result["value"].diff().dropna().unique()
    assert all(d == 10 for d in diffs), f"间隔应恒为10, 实际: {diffs}"
    print(f"[PASS] 随机偏移采样: 起点={offset}, 样本数={len(result)}")


def test_series_sampling():
    s = pd.Series(range(50), name="num")
    result = equidistant_sample(s, interval=5, random_offset=False, seed=0)
    expected = s.iloc[::5].reset_index(drop=True)
    assert result.equals(expected)
    print(f"[PASS] Series采样: 50条 → {len(result)}条")


def test_small_dataset():
    df = pd.DataFrame({"value": range(3)})
    result = equidistant_sample(df, interval=10, random_offset=False, seed=0)
    assert len(result) == 1
    assert result["value"].iloc[0] == 0
    print(f"[PASS] 小数据集: 3条 interval=10 → {len(result)}条")


def test_empty_dataset():
    df = pd.DataFrame({"value": pd.Series(dtype=int)})
    result = equidistant_sample(df, interval=10, random_offset=False, seed=0)
    assert len(result) == 0
    print("[PASS] 空数据集: 返回空")


def test_interval_1():
    df = pd.DataFrame({"value": range(5)})
    result = equidistant_sample(df, interval=1, random_offset=False, seed=0)
    assert len(result) == 5
    print(f"[PASS] interval=1: 返回全部 {len(result)}条")


def test_invalid_interval():
    try:
        equidistant_sample(pd.DataFrame({"a": [1]}), interval=0)
        assert False, "应抛出异常"
    except ValueError as e:
        print(f"[PASS] 无效interval: {e}")


def test_sampler_class():
    sampler = EquidistantSampler(interval=10, random_offset=True, seed=7)
    df = pd.DataFrame({"value": range(200)})
    r1 = sampler.sample(df)
    offset1 = sampler.last_offset
    sampler2 = EquidistantSampler(interval=10, random_offset=True, seed=7)
    r2 = sampler2.sample(df)
    assert r1.equals(r2), "相同seed应产生相同结果"
    assert sampler2.last_offset == offset1
    print(f"[PASS] Sampler类: offset={offset1}, 样本数={len(r1)}")


def test_reproducibility():
    df = pd.DataFrame({"value": range(100)})
    r1 = equidistant_sample(df, interval=7, random_offset=True, seed=123)
    r2 = equidistant_sample(df, interval=7, random_offset=True, seed=123)
    assert r1.equals(r2), "相同seed应产生相同结果"
    print(f"[PASS] 可复现性: seed=123 → {len(r1)}条")


if __name__ == "__main__":
    test_basic_sampling()
    test_random_offset()
    test_series_sampling()
    test_small_dataset()
    test_empty_dataset()
    test_interval_1()
    test_invalid_interval()
    test_sampler_class()
    test_reproducibility()
    print("\n全部测试通过!")
