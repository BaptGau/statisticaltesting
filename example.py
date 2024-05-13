import numpy as np
from src.two_samples_tests import (
    StudentTest,
    MannUWhitneyTest,
    KolmogorovSmirnovTest,
    BartlettTest,
    LeveneTest,
)
from src.one_sample_tests import ShapiroWilkTest

# Parametric central tendancy test (assumption: samples have the same variance)
sample1 = np.random.normal(loc=10, scale=3, size=100)
sample2 = np.random.normal(loc=9.3, scale=3, size=100)

tester = StudentTest()
tester.fit(
    np.random.normal(loc=10, scale=3, size=100),
    np.random.normal(loc=9.3, scale=4.5, size=100),
    plot_results=True,
)
print(tester.params)

# Non-parametric central tendancy test
tester = MannUWhitneyTest()
tester.fit(sample1, sample2, plot_results=True)
print(tester.params)

# Non-parametric dispersion test
sample1 = np.random.gamma(shape=5, scale=10, size=100)
sample2 = np.random.gamma(shape=3, scale=12, size=100)

tester = LeveneTest()
tester.fit(sample1, sample2, plot_results=True)
print(tester.params)

# Parametric dispersion test (assumption: samples are not skewed)
tester = BartlettTest()
tester.fit(sample1, sample2, plot_results=True)
print(tester.params)

# Distribution test, often used for goodness of fit between two samples
tester = KolmogorovSmirnovTest()
tester.fit(sample1, sample2, plot_results=True)
print(tester.params)

# Distribution test, determine if sample is gaussian
tester = ShapiroWilkTest()
tester.fit(sample1, plot_results=True)
print(tester.params)
