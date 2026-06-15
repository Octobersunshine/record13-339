import pandas as pd
import numpy as np
import warnings
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
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = equidistant_sample(df, interval=10, random_offset=False, seed=0)
        assert len(w) == 1, f"应发出1条警告, 实际 {len(w)}条"
        assert "大于数据总量" in str(w[0].message)
        assert issubclass(w[0].category, UserWarning)
    assert len(result) == 3, f"interval>n时应返回全部数据, 实际 {len(result)}条"
    assert result["value"].tolist() == [0, 1, 2]
    print(f"[PASS] 小数据集: 3条 interval=10 → 返回全部 {len(result)}条, 且发出警告")


def test_interval_equal_to_n():
    df = pd.DataFrame({"value": range(5)})
    result = equidistant_sample(df, interval=5, random_offset=False, seed=0)
    assert len(result) == 1
    assert result["value"].iloc[0] == 0
    print(f"[PASS] interval==n: 5条 interval=5 → {len(result)}条")


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


def test_sampler_class_small_dataset():
    sampler = EquidistantSampler(interval=100, random_offset=True, seed=1)
    df = pd.DataFrame({"value": range(5)})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = sampler.sample(df)
        assert len(w) == 1
        assert "大于数据总量" in str(w[0].message)
    assert len(result) == 5
    assert sampler.last_offset is None
    print(f"[PASS] Sampler类小数据集: 5条 interval=100 → 返回全部, last_offset=None")


def test_circular_basic():
    df = pd.DataFrame({"value": range(10)})
    result = equidistant_sample(
        df, interval=3, random_offset=False, circular=True, n_samples=5
    )
    expected_values = [0, 3, 6, 9, 2]
    assert result["value"].tolist() == expected_values, (
        f"环形采样失败: {result['value'].tolist()} != {expected_values}"
    )
    print(f"[PASS] 环形采样基础: n=10,interval=3,n_samples=5 → {result['value'].tolist()}")


def test_circular_default_n_samples():
    df = pd.DataFrame({"value": range(10)})
    result = equidistant_sample(
        df, interval=3, random_offset=False, circular=True
    )
    expected_count = int(np.ceil(10 / 3))
    assert len(result) == expected_count
    print(f"[PASS] 环形采样默认数量: 默认 n_samples={expected_count}")


def test_circular_large_interval():
    df = pd.DataFrame({"value": range(5)})
    result = equidistant_sample(
        df, interval=10, random_offset=False, circular=True, n_samples=4
    )
    expected_values = [0, 0, 0, 0]
    assert result["value"].tolist() == expected_values, (
        f"大间隔环形采样失败: {result['value'].tolist()} != {expected_values}"
    )
    print(f"[PASS] 大间隔环形采样: interval=10,n=5,n_samples=4 → {result['value'].tolist()}")


def test_circular_random_offset_seed():
    df = pd.DataFrame({"value": range(10)})
    r1 = equidistant_sample(
        df, interval=3, random_offset=True, circular=True, n_samples=6, seed=7
    )
    r2 = equidistant_sample(
        df, interval=3, random_offset=True, circular=True, n_samples=6, seed=7
    )
    assert r1.equals(r2)
    offset = r1["value"].iloc[0]
    step = (r1["value"].iloc[1] - offset) % 10
    assert step == 3, f"环形间隔应为3 mod 10, 实际: {step}"
    print(f"[PASS] 环形随机偏移+seed: offset={offset}, 样本={r1['value'].tolist()}")


def test_circular_sampler_class():
    sampler = EquidistantSampler(
        interval=4, random_offset=False, circular=True, n_samples=6
    )
    df = pd.DataFrame({"value": range(7)})
    result = sampler.sample(df)
    expected_values = [0, 4, 1, 5, 2, 6]
    assert result["value"].tolist() == expected_values
    print(f"[PASS] Sampler类环形采样: {result['value'].tolist()}")


def test_n_samples_truncation():
    df = pd.DataFrame({"value": range(100)})
    result = equidistant_sample(
        df, interval=10, random_offset=False, circular=False, n_samples=3
    )
    assert result["value"].tolist() == [0, 10, 20]
    print(f"[PASS] n_samples截断: {result['value'].tolist()}")


def test_invalid_n_samples():
    try:
        equidistant_sample(pd.DataFrame({"a": [1, 2]}), n_samples=0)
        assert False, "应抛出异常"
    except ValueError as e:
        print(f"[PASS] 无效n_samples: {e}")


if __name__ == "__main__":
    test_basic_sampling()
    test_random_offset()
    test_series_sampling()
    test_small_dataset()
    test_interval_equal_to_n()
    test_empty_dataset()
    test_interval_1()
    test_invalid_interval()
    test_sampler_class()
    test_sampler_class_small_dataset()
    test_reproducibility()
    test_circular_basic()
    test_circular_default_n_samples()
    test_circular_large_interval()
    test_circular_random_offset_seed()
    test_circular_sampler_class()
    test_n_samples_truncation()
    test_invalid_n_samples()
    print("\n全部测试通过!")
