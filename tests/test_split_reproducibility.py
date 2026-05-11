from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


def test_train_test_split_is_deterministic_with_fixed_random_state() -> None:
    features = np.arange(120).reshape(60, 2)
    target = np.arange(60)

    split_a = train_test_split(features, target, test_size=0.2, random_state=42, shuffle=True)
    split_b = train_test_split(features, target, test_size=0.2, random_state=42, shuffle=True)

    for array_a, array_b in zip(split_a, split_b, strict=True):
        assert np.array_equal(array_a, array_b)


def test_train_test_split_changes_with_different_random_state() -> None:
    features = np.arange(120).reshape(60, 2)
    target = np.arange(60)

    split_a = train_test_split(features, target, test_size=0.2, random_state=42, shuffle=True)
    split_b = train_test_split(features, target, test_size=0.2, random_state=43, shuffle=True)

    assert not np.array_equal(split_a[1], split_b[1])
