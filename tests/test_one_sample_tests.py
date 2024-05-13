import numpy as np
from scipy.stats import shapiro
import pytest

from src.base import TestParams
from src.one_sample_tests import ShapiroWilkTest


@pytest.fixture
def test_objects():
    return [ShapiroWilkTest()]


def test_params_type(test_objects):
    for mock_object in test_objects:
        assert isinstance(
            mock_object.params, TestParams
        ), f"{mock_object} is not using the TestParams dataclass"


# Distribution tests
def test_shapiro_test_pass():
    mock = np.random.normal(loc=5, scale=10, size=100)
    res_expected = shapiro(mock)
    shapiro_res = ShapiroWilkTest().fit(mock)

    assert np.isclose(
        res_expected[0], shapiro_res.params.test_statistic, atol=0.01
    ), "tests statistic does not match reference"
    assert np.isclose(
        res_expected[1], shapiro_res.params.test_pval, atol=0.01
    ), "tests pval does not match reference"
    assert (
        shapiro_res.params.test_pval > 0.05
    ), "tests conclusion does not match expectation"


def test_shapiro_test_does_not_pass():
    mock = np.random.uniform(low=0, high=10, size=100)
    res_expected = shapiro(mock)
    shapiro_res = ShapiroWilkTest().fit(mock)

    assert np.isclose(
        res_expected[0], shapiro_res.params.test_statistic, atol=0.01
    ), "tests statistic does not match reference"
    assert np.isclose(
        res_expected[1], shapiro_res.params.test_pval, atol=0.01
    ), "tests pvalue does not match reference"
    assert (
        shapiro_res.params.test_pval <= 0.05
    ), "tests conclusion does not match expectation"
