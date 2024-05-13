from abc import ABC, abstractmethod
from dataclasses import dataclass
from numpy._typing import ArrayLike


@dataclass
class TestParams:
    test_name: str = None
    test_statistic: float = None
    test_pval: float = None
    test_h0: str = None
    is_fitted: bool = False


class OneSampleTest(ABC):
    params = TestParams()

    def fit(self, x: ArrayLike, plot_results: bool = False) -> "OneSampleTest":
        self._compute_test(x)
        if plot_results:
            self._plot_results(x)
        return self

    @abstractmethod
    def _compute_test(self, x: ArrayLike) -> None:
        pass

    @abstractmethod
    def _plot_results(self, x: ArrayLike) -> None:
        pass


class TwoSampleTest(ABC):
    params = TestParams()

    def fit(
        self, x: ArrayLike, y: ArrayLike, plot_results: bool = False
    ) -> "TwoSampleTest":
        self._compute_test(x, y)
        if plot_results:
            self._plot_results(x, y)
        return self

    @abstractmethod
    def _compute_test(self, x: ArrayLike, y: ArrayLike) -> None:
        pass

    @abstractmethod
    def _plot_results(self, x: ArrayLike, y: ArrayLike) -> None:
        pass
