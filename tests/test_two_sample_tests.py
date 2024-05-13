import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu, kstest, bartlett, levene
import pytest

from src.two_samples_tests import (
    StudentTest,
    MannUWhitneyTest,
    KolmogorovSmirnovTest,
    LeveneTest,
    BartlettTest,
)
from src.base import TestParams


@pytest.fixture
def test_objects():
    return [
        StudentTest(),
        MannUWhitneyTest(),
        LeveneTest(),
        BartlettTest(),
        KolmogorovSmirnovTest(),
    ]


def test_params_type(test_objects):
    for mock_object in test_objects:
        assert isinstance(
            mock_object.params, TestParams
        ), f"{mock_object} is not using the TestParams dataclass"


# Central tendancy tests
def test_student_test_pass():
    mock_1 = np.random.normal(loc=5, scale=10, size=100)
    mock_2 = np.random.normal(loc=5.1, scale=10, size=100)
    res_expected = ttest_ind(mock_1, mock_2)
    student_res = StudentTest().fit(mock_1, mock_2)

    assert np.isclose(
        res_expected[0], student_res.params.test_statistic, atol=0.01
    ), "tests statistic does not match reference"
    assert np.isclose(
        res_expected[1], student_res.params.test_pval, atol=0.01
    ), "tests pval does not match reference"
    assert (
        student_res.params.test_pval > 0.01
    ), "tests conclusion does not match expectation"


def test_student_test_does_not_pass():
    mock_1 = np.random.normal(loc=10, scale=10, size=100)
    mock_2 = np.random.normal(loc=5.1, scale=10, size=100)
    res_expected = ttest_ind(mock_1, mock_2)
    student_res = StudentTest().fit(mock_1, mock_2)

    assert np.isclose(
        res_expected[0], student_res.params.test_statistic, atol=0.01
    ), "tests statistic does not match reference"
    assert np.isclose(
        res_expected[1], student_res.params.test_pval, atol=0.01
    ), "tests pvalue does not match reference"
    assert (
        student_res.params.test_pval <= 0.05
    ), "tests conclusion does not match expectation"


def test_muw_test_pass():
    mock_1 = np.random.normal(loc=5, scale=10, size=100)
    mock_2 = np.random.normal(loc=5.1, scale=10, size=100)
    res_expected = mannwhitneyu(mock_1, mock_2)
    muw_res = MannUWhitneyTest().fit(mock_1, mock_2)

    assert np.isclose(
        res_expected[0], muw_res.params.test_statistic, atol=0.01
    ), "tests statistic does not match reference"
    assert np.isclose(
        res_expected[1], muw_res.params.test_pval, atol=0.01
    ), "tests pval does not match reference"
    assert (
        muw_res.params.test_pval > 0.05
    ), "tests conclusion does not match expectation"


def test_muw_test_does_not_pass():
    mock_1 = np.random.normal(loc=10, scale=10, size=100)
    mock_2 = np.random.normal(loc=5.1, scale=10, size=100)
    res_expected = mannwhitneyu(mock_1, mock_2)
    muw_res = MannUWhitneyTest().fit(mock_1, mock_2)

    assert np.isclose(
        res_expected[0], muw_res.params.test_statistic, atol=0.01
    ), "tests statistic does not match reference"
    assert np.isclose(
        res_expected[1], muw_res.params.test_pval, atol=0.01
    ), "tests pval does not match reference"
    assert (
        muw_res.params.test_pval <= 0.01
    ), "tests conclusion does not match expectation"


# Dispersion tests
def test_bartlett_test_pass():
    mock_1 = np.random.normal(loc=3, scale=10, size=1000)
    mock_2 = np.random.normal(loc=6, scale=10, size=1000)
    res_expected = bartlett(mock_1, mock_2)
    bartlett_res = BartlettTest().fit(mock_1, mock_2)

    assert np.isclose(
        res_expected[0], bartlett_res.params.test_statistic, atol=0.01
    ), "tests statistic does not match reference"
    assert np.isclose(
        res_expected[1], bartlett_res.params.test_pval, atol=0.01
    ), "tests pval does not match reference"


def test_bartlett_test_does_not_pass():
    mock_1 = np.random.normal(loc=3, scale=1, size=1000)
    mock_2 = np.random.normal(loc=7, scale=10, size=1000)
    res_expected = bartlett(mock_1, mock_2)
    bartlett_res = BartlettTest().fit(mock_1, mock_2)

    assert np.isclose(
        res_expected[0], bartlett_res.params.test_statistic, atol=0.01
    ), "tests statistic does not match reference"
    assert np.isclose(
        res_expected[1], bartlett_res.params.test_pval, atol=0.01
    ), "tests pval does not match reference"
    assert (
        bartlett_res.params.test_pval <= 0.01
    ), "tests conclusion does not match expectation"


def test_levene_test_pass():
    mock_1 = np.random.normal(loc=6, scale=10, size=1000)
    mock_2 = np.random.normal(loc=6, scale=10, size=1000)
    res_expected = levene(mock_1, mock_2, center="median")
    levene_res = LeveneTest().fit(mock_1, mock_2)

    assert np.isclose(
        res_expected[0], levene_res.params.test_statistic, atol=200
    ), "tests statistic does not match reference"


def test_levene_test_does_not_pass():
    mock_1 = np.random.normal(loc=3, scale=1, size=1000)
    mock_2 = np.random.uniform(low=2, high=15, size=1000)
    res_expected = levene(mock_1, mock_2)
    levene_res = LeveneTest().fit(mock_1, mock_2)

    assert np.isclose(
        res_expected[0], levene_res.params.test_statistic, atol=200
    ), "tests statistic does not match reference"
    assert np.isclose(
        res_expected[1], levene_res.params.test_pval, atol=0.01
    ), "tests pval does not match reference"


# Distribution tests
def test_ks_test_pass():
    mock_1 = np.random.normal(loc=5, scale=10, size=1000)
    mock_2 = np.random.normal(loc=5.1, scale=10, size=1000)
    res_expected = kstest(mock_1, mock_2)
    ks_res = KolmogorovSmirnovTest().fit(mock_1, mock_2)

    assert np.isclose(
        res_expected[0], ks_res.params.test_statistic, atol=0.01
    ), "tests statistic does not match reference"
    assert np.isclose(
        res_expected[1], ks_res.params.test_pval, atol=0.01
    ), "tests pval does not match reference"
    assert ks_res.params.test_pval > 0.05, "tests conclusion does not match expectation"


def test_ks_does_not_pass():
    mock_1 = np.random.normal(loc=10, scale=10, size=1000)
    mock_2 = np.random.normal(loc=5.1, scale=10, size=1000)
    res_expected = kstest(mock_1, mock_2)
    ks_res = KolmogorovSmirnovTest().fit(mock_1, mock_2)

    assert np.isclose(
        res_expected[0], ks_res.params.test_statistic, atol=0.01
    ), "tests statistic does not match reference"
    assert np.isclose(
        res_expected[1], ks_res.params.test_pval, atol=0.01
    ), "tests pval does not match reference"
    assert (
        ks_res.params.test_pval <= 0.01
    ), "tests conclusion does not match expectation"
