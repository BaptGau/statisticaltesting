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
    """
    Class associated to the Student's "T" test, used to compare the means of two distributions (assuming variances are equal).
    For more details: https://en.wikipedia.org/wiki/Student%27s_t-test.
    """

    def __init__(self):
        super().__init__()
        self.params.test_h0 = r"$\mu_0 = \mu_1$"
        self.params.test_name = "Student test"

    def _compute_test(self, x: ArrayLike, y: ArrayLike) -> None:
        """
        Run the Student's T test between the two samples.

        Parameters
        ----------
        x (ArrayLike): The first sample to be tested.
        y (ArrayLike): The second sample to be tested.
        """
        t_test_results = ttest_ind(x, y, equal_var=True)
        self.params.test_statistic = t_test_results[0]
        self.params.test_pval = t_test_results[1]
        self.params.is_fitted = True

    def _plot_results(self, x: ArrayLike, y: ArrayLike) -> None:
        """
        The method that will plot the result of the Student's test.

        Parameters
        ----------
        x (ArrayLike): The first sample to be tested.
        y (ArrayLike): The second sample to be tested.
        """
        plot_results_student_test(x=x, y=y, results=self.params)


class MannUWhitneyTest(TwoSampleTest):
    """
    Class associated to the Mann Wilcoxon Whitney's (or 2 samples Kruskall-Wallis) "U" test, used to compare the medians of two distributions.
    Can be seen as a non-parametric version of the Student's T test. For more details: https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test.
    """
    def __init__(self):
        super().__init__()
        self.params.test_h0 = r"P(X > Y) = P(Y > X)"
        self.params.test_name = "Mann Wilcoxon Whitney U Test"

    def _compute_test(self, x: ArrayLike, y: ArrayLike) -> None:
        """
        Run the MWW U test between the two samples.

        Parameters
        ----------
        x (ArrayLike): The first sample to be tested.
        y (ArrayLike): The second sample to be tested.
       """
        muw_results = mannwhitneyu(x, y, use_continuity=True)
        self.params.test_statistic = muw_results[0]
        self.params.test_pval = muw_results[1]
        self.params.is_fitted = True

    def _plot_results(self, x: ArrayLike, y: ArrayLike) -> None:
        """
        The method that will plot the result of the MWW's test.

        Parameters
        ----------
        x (ArrayLike): The first sample to be tested.
        y (ArrayLike): The second sample to be tested.
        """
        plot_results_standard_test(x=x, y=y, results=self.params)


# Dispersion tests
class LeveneTest(TwoSampleTest):
    """
    Class associated to the Levene's test, used to compare the variances of two distributions.
    Less sensitive to the non-normality than Bartlett's test. For more details: https://en.wikipedia.org/wiki/Levene%27s_test.
    """
    def __init__(self):
        super().__init__()
        self.params.test_h0 = r"$\sigma^2_1 = \sigma^2_2$"
        self.params.test_name = "Levene Test"

    def _compute_test(self, x: ArrayLike, y: ArrayLike) -> None:
        """
        Run the Levene's test between the two samples.

        Parameters
        ----------
        x (ArrayLike): The first sample to be tested.
        y (ArrayLike): The second sample to be tested.
        """
        levene_results = levene(x, y, center="median")
        self.params.test_statistic = levene_results[0]
        self.params.test_pval = levene_results[1]
        self.params.is_fitted = True

    def _plot_results(self, x: ArrayLike, y: ArrayLike) -> None:
        """
        The method that will plot the result of the Levene's test.

        Parameters
        ----------
        x (ArrayLike): The first sample to be tested.
        y (ArrayLike): The second sample to be tested.
        """
        plot_results_standard_test(x=x, y=y, results=self.params)


class BartlettTest(TwoSampleTest):
    """
    Class associated to the Bartlett's test, used to compare the variances of two distributions.
    For more details: https://en.wikipedia.org/wiki/Bartlett%27s_test.
    """
    def __init__(self):
        super().__init__()
        self.params.test_h0 = r"$\sigma^2_1 = \sigma^2_2$"
        self.params.test_name = "Bartlett test"

    def _compute_test(self, x: ArrayLike, y: ArrayLike) -> None:
        """
        Run the Bartlett's test between the two samples.

        Parameters
        ----------
        x (ArrayLike): The first sample to be tested.
        y (ArrayLike): The second sample to be tested.
        """
        bartlett_results = bartlett(x, y)
        self.params.test_statistic = bartlett_results[0]
        self.params.test_pval = bartlett_results[1]
        self.params.is_fitted = True

    def _plot_results(self, x: ArrayLike, y: ArrayLike) -> None:
        """
        The method that will plot the result of the Levene's test.

        Parameters
        ----------
        x (ArrayLike): The first sample to be tested.
        y (ArrayLike): The second sample to be tested.
        """
        plot_results_standard_test(x=x, y=y, results=self.params)


# Distribution tests
class KolmogorovSmirnovTest(TwoSampleTest):
    """
    Class associated to the Kolmogorov-Smirnov's test, used to compare two distributions (or the goodness of fit between one and another).
    Samples need to be big enough for the test to be relevant. For more details: https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test.
    """
    def __init__(self):
        super().__init__()
        self.params.test_name = "Kolmogorov Smirnov Test"
        self.params.test_h0 = r"The two distributions are the same"

    def _compute_test(self, x: ArrayLike, y: ArrayLike) -> None:
        """
        Run the Kolmogorov-Smirnov's test between the two samples.

        Parameters
        ----------
        x (ArrayLike): The first sample to be tested.
        y (ArrayLike): The second sample to be tested.
        """
        ks_results = kstest(rvs=x, cdf=y)
        self.params.test_statistic = ks_results[0]
        self.params.test_pval = ks_results[1]
        self.params.is_fitted = True

    def _plot_results(self, x: ArrayLike, y: ArrayLike) -> None:
        """
        The method that will plot the result of the Kolmogorov-Smirnov's test.

        Parameters
        ----------
        x (ArrayLike): The first sample to be tested.
        y (ArrayLike): The second sample to be tested.
       """
        plot_results_ks_test(x, y, self.params)
