import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from scipy.stats import t
import mplcyberpunk as mpl
import seaborn as sns

from src.base import TestParams

plt.style.use("cyberpunk")
COLORS = ["dodgerblue", "lime"]


# tools
def __kde_plotter(x: ArrayLike, y: ArrayLike, ax, **kwargs) -> None:
    x_name = kwargs.get("x_name", "First sample")
    y_name = kwargs.get("y_name", "Second sample")
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    sns.kdeplot(x=x, label=f"KDE of {x_name}", ax=ax, color=COLORS[0])
    ax.axvline(
        mean_x,
        color=COLORS[0],
        lw=1,
        linestyle="--",
        label=f"{x_name} mean| " + r"$\mu=$" + f"{mean_x:.2f}",
    )
    sns.kdeplot(x=y, label=f"KDE of {y_name}", ax=ax, color=COLORS[1])
    ax.axvline(
        np.mean(y),
        color=COLORS[1],
        lw=1,
        linestyle="--",
        label=f"{y_name} mean | " + r"$\mu=$" + f"{mean_y:.2f}",
    )
    ax.set_title(f"Kernel Density Estimations", fontweight="bold")
    ax.set_xlabel("Data", fontweight="bold")
    ax.set_ylabel("Density", fontweight="bold")
    ax.legend()
    ax.grid(True)
    mpl.add_gradient_fill(ax=ax, alpha_gradientglow=0.3)


def __violin_plotter(x: ArrayLike, y: ArrayLike, ax, **kwargs) -> None:
    x_name = kwargs.get("x_name", "First sample")
    y_name = kwargs.get("y_name", "Second sample")

    violins = ax.violinplot(
        [x, y], vert=True, showmeans=False, showmedians=True  # vertical box alignment
    )
    ax.set_xticks(
        [y + 1 for y in range(2)],
        labels=[
            rf"{x_name} | $|\sigma:$" + f"{np.std(x):.2f}",
            f"{y_name} | $|\sigma:$" + f"{np.std(y):.2f}",
        ],
        fontweight="bold",
    )
    for i, pc in enumerate(violins["bodies"]):
        pc.set_facecolor(COLORS[i])
        pc.set_edgecolor("white")
    violins["cbars"].set_color(COLORS)
    violins["cmedians"].set_color(COLORS)
    violins["cmaxes"].set_color(COLORS)
    violins["cmins"].set_color(COLORS)


def __ecdf_plotter(x: ArrayLike, y: ArrayLike, ax, **kwargs) -> None:
    x_name = kwargs.get("x_name", "First sample")
    y_name = kwargs.get("y_name", "Second sample")

    sns.ecdfplot(x=x, label=f"Empirical CDF of {x_name}", ax=ax, color=COLORS[0])
    sns.ecdfplot(x=y, label=f"Empirical CDF of {y_name}", ax=ax, color=COLORS[1])
    ax.set_title(f"Empirical Cumulative Distribution Functions", fontweight="bold")
    ax.set_xlabel("Data", fontweight="bold")
    ax.set_ylabel("Cumulative Probability", fontweight="bold")
    ax.legend()
    ax.grid(True)


def __student_plotter(
    x: ArrayLike, y: ArrayLike, student_result: TestParams, ax
) -> None:
    ax.axvline(
        x=student_result.test_statistic,
        color="orangered",
        linestyle="--",
        label="t-statistic",
    )

    x_shade = np.linspace(student_result.test_statistic, 5, 100)
    ax.fill_between(
        x=np.linspace(-5, student_result.test_statistic, 100),
        y1=0,
        y2=t.pdf(
            np.linspace(-5, student_result.test_statistic, 100),
            df=len(x) + len(y) - 1,
        ),
        color="dodgerblue",
        label="T-distribution",
        alpha=0.3,
    )

    ax.fill_between(
        x=x_shade,
        y1=0,
        y2=t.pdf(x_shade, df=len(x) + len(y) - 2),
        color="orangered",
        alpha=0.3,
        label="p-value area",
    )

    ax.text(
        student_result.test_statistic + 0.1,
        0.2,
        r"$P(X|$" + student_result.test_h0 + f"$) = {student_result.test_pval:.3f}$",
        color="red",
    )
    ax.set_xlabel("t-value", fontweight="bold")
    ax.set_ylabel("Density", fontweight="bold")
    ax.set_title("Student's t-test", fontweight="bold")
    ax.legend()


def __add_suptitle(fig, results: TestParams):
    fig.suptitle(
        f"{results.test_name} - P(X|{results.test_h0}): {results.test_pval:.2f}",
        fontweight="bold",
    )


# Central tendancy analysis
def plot_results_student_test(x: ArrayLike, y: ArrayLike, results: TestParams) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    __kde_plotter(x=x, y=y, ax=axs[0])
    __student_plotter(x=x, y=y, student_result=results, ax=axs[1])
    __add_suptitle(fig=fig, results=results)
    plt.show()


def plot_results_standard_test(x: ArrayLike, y: ArrayLike, results: TestParams) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    __kde_plotter(x=x, y=y, ax=axs[0])
    __violin_plotter(x=x, y=y, ax=axs[1])
    __add_suptitle(fig=fig, results=results)
    plt.show()


# Distribution analysis
def plot_results_ks_test(x: ArrayLike, y: ArrayLike, ks_results: TestParams) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    __kde_plotter(x, y, axs[0])
    __ecdf_plotter(x, y, axs[1])
    __add_suptitle(fig=fig, results=ks_results)
    plt.show()


def plot_results_shapiro_test(x: ArrayLike, shapiro_results: TestParams) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    x_name = "Sample"
    y_name = "Inferred Normal Distribution"
    normal_sample = np.random.normal(loc=np.mean(x), scale=np.std(x), size=1000)
    __kde_plotter(
        x=x,
        y=normal_sample,
        ax=axs[0],
        x_name=x_name,
        y_name=y_name,
    )
    __ecdf_plotter(
        x=x,
        y=normal_sample,
        ax=axs[1],
        x_name=x_name,
        y_name=y_name,
    )
    __add_suptitle(fig=fig, results=shapiro_results)
    plt.show()
