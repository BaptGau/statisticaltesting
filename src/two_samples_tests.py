from abc import ABC, abstractmethod
from numpy.typing import ArrayLike
from scipy.stats import ttest_ind, mannwhitneyu, kstest, levene, bartlett

from src.base import TwoSampleTest
from src.plotter import (
    plot_results_student_test,
    plot_results_ks_test,
    plot_results_standard_test,
)


# Central Tendancy Test
class StudentTest(TwoSampleTest):
    def __init__(self):
        super().__init__()
        self.params.test_h0 = r"$\mu_0 = \mu_1$"
        self.params.test_name = "Student tests"

    def _compute_test(self, x: ArrayLike, y: ArrayLike) -> None:
        t_test_results = ttest_ind(x, y, equal_var=False)
        self.params.test_statistic = t_test_results[0]
        self.params.test_pval = t_test_results[1]
        self.params.is_fitted = True

    def _plot_results(self, x: ArrayLike, y: ArrayLike) -> None:
        plot_results_student_test(x=x, y=y, results=self.params)


class MannUWhitneyTest(TwoSampleTest):
    def __init__(self):
        super().__init__()
        self.params.test_h0 = r"P(X > Y) = P(Y > X)"
        self.params.test_name = "Mann Wilcoxon Whitney U Test"

    def _compute_test(self, x: ArrayLike, y: ArrayLike) -> None:
        muw_results = mannwhitneyu(x, y, use_continuity=True)
        self.params.test_statistic = muw_results[0]
        self.params.test_pval = muw_results[1]
        self.params.is_fitted = True

    def _plot_results(self, x: ArrayLike, y: ArrayLike) -> None:
        plot_results_standard_test(x=x, y=y, results=self.params)


# Dispersion tests
class LeveneTest(TwoSampleTest):
    def __init__(self):
        super().__init__()
        self.params.test_h0 = r"$\sigma^2_1 = \sigma^2_2$"
        self.params.test_name = "Levene Test"

    def _compute_test(self, x: ArrayLike, y: ArrayLike) -> None:
        levene_results = levene(x, y, center="median")
        self.params.test_statistic = levene_results[0]
        self.params.test_pval = levene_results[1]
        self.params.is_fitted = True

    def _plot_results(self, x: ArrayLike, y: ArrayLike) -> None:
        plot_results_standard_test(x=x, y=y, results=self.params)


class BartlettTest(TwoSampleTest):
    def __init__(self):
        super().__init__()
        self.params.test_h0 = r"$\sigma^2_1 = \sigma^2_2$"
        self.params.test_name = "Bartlett test"

    def _compute_test(self, x: ArrayLike, y: ArrayLike) -> None:
        bartlett_results = bartlett(x, y)
        self.params.test_statistic = bartlett_results[0]
        self.params.test_pval = bartlett_results[1]
        self.params.is_fitted = True

    def _plot_results(self, x: ArrayLike, y: ArrayLike) -> None:
        plot_results_standard_test(x=x, y=y, results=self.params)


# Distribution tests
class KolmogorovSmirnovTest(TwoSampleTest):
    def __init__(self):
        super().__init__()
        self.params.test_name = "Kolmogorov Smirnov Test"
        self.params.test_h0 = r"The two distributions are the same"

    def _compute_test(self, x: ArrayLike, y: ArrayLike) -> None:
        ks_results = kstest(rvs=x, cdf=y)
        self.params.test_statistic = ks_results[0]
        self.params.test_pval = ks_results[1]
        self.params.is_fitted = True

    def _plot_results(self, x: ArrayLike, y: ArrayLike) -> None:
        plot_results_ks_test(x, y, self.params)
