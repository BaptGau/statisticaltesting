from abc import ABC, abstractmethod
from dataclasses import dataclass
from numpy._typing import ArrayLike


@dataclass
class TestParams:
    """
    Dataclass used to store test results. Attributes are:
    test_name (str)
    test_statistic (float)
    test_pval (float)
    test_h0 (str)
    is_fitted (bool)
    """

    test_name: str = None
    test_statistic: float = None
    test_pval: float = None
    test_h0: str = None
    is_fitted: bool = False


class OneSampleTest(ABC):
    """
    Abstract class that define the interface for the one sample test (Shapiro-Wilk, Anderson-Darling, ...)
    """

    def __init__(self):
        self.params = TestParams()

    def fit(self, x: ArrayLike, plot_results: bool = False) -> "OneSampleTest":
        """
        Based on the sklearn API, run the test.

        Parameters
        ----------
        x (ArrayLike): The sample to be tested.
        plot_results (bool): Wheither to plot the results of the test. Defaults to False.

        Returns
        -------
        The fitted object.
        """
        self._compute_test(x)
        if plot_results:
            self._plot_results(x)
        return self

    @abstractmethod
    def _compute_test(self, x: ArrayLike) -> None:
        """
        The method where the test will be run in.

        Parameters
        ----------
        x (ArrayLike): The sample to be tested.
        """
        pass

    @abstractmethod
    def _plot_results(self, x: ArrayLike) -> None:
        """
        The method that will plot the result of the test.

        Parameters
        ----------
        x (ArrayLike): The sample to be tested.
        """
        pass


class TwoSampleTest(ABC):
    """
    Abstract class that define the interface for the two samples tests (Student, Kruskall-Wallis, Levene, Kolmogorov-Smirnov, ...)
    """
    def __init__(self):
        self.params = TestParams()

    def fit(
        self, x: ArrayLike, y: ArrayLike, plot_results: bool = False
    ) -> "TwoSampleTest":
        """
        Based on the sklearn API, run the test.

        Parameters
        ----------
        x (ArrayLike): The first sample to be tested.
        y (ArrayLike): The second sample to be tested.
        plot_results (bool): Wheither to plot the results of the test. Defaults to False.

        Returns
        -------
        The fitted object.
        """
        self._compute_test(x, y)
        if plot_results:
            self._plot_results(x, y)
        return self

    @abstractmethod
    def _compute_test(self, x: ArrayLike, y: ArrayLike) -> None:
        """
        The method where the test will be run in.

        Parameters
        ----------
        x (ArrayLike): The first sample to be tested.
        y (ArrayLike): The second sample to be tested.
        """
        pass

    @abstractmethod
    def _plot_results(self, x: ArrayLike, y: ArrayLike) -> None:
        """
        The method that will plot the result of the test.

        Parameters
        ----------
        x (ArrayLike): The first sample to be tested.
        y (ArrayLike): The second sample to be tested.
        """
        pass
