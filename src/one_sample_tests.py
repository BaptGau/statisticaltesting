from numpy._typing import ArrayLike
from scipy.stats import shapiro, anderson

from src.base import OneSampleTest
from src.plotter import plot_results_shapiro_test


class ShapiroWilkTest(OneSampleTest):
    def __init__(self):
        super().__init__()
        self.params.test_h0 = r"$X~N(.)$"
        self.params.test_name = "Shapiro Wilk test"

    def _compute_test(self, x: ArrayLike) -> None:
        shapiro_results = shapiro(x)
        self.params.test_statistic = shapiro_results[0]
        self.params.test_pval = shapiro_results[1]
        self.params.is_fitted = True

    def _plot_results(self, x: ArrayLike) -> None:
        plot_results_shapiro_test(x, shapiro_results=self.params)
