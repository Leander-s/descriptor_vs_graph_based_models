import numpy as np
from scipy.stats import friedmanchisquare, rankdata, t


def demsar(results):
    """
    Performs a full Demšar test suite:
    - Average ranks computation
    - Friedman test
    - Iman-Davenport correction
    - Bonferroni-Dunn post hoc test

    Parameters:
        results: np.ndarray
            A 2D array where rows are datasets and columns are algorithms.

    Returns:
        dict: Results of all tests, including average ranks, p-values, and post hoc comparisons.
    """
    avg_ranks = get_average_ranks(results)
    friedman_stat, friedman_p = friedman(results)
    iman_davenport_stat, iman_davenport_p = iman_davenport(
        results, friedman_stat)
    post_hoc_results = bonferroni(avg_ranks, results.shape[1])

    return {
        "average_ranks": avg_ranks.round(2),
        "friedman": {"statistic": friedman_stat, "p_value": friedman_p},
        "iman_davenport": {"statistic": iman_davenport_stat, "p_value": iman_davenport_p},
        "bonferroni": post_hoc_results,
    }


def get_average_ranks(results):
    """
    Computes the average ranks for each algorithm across datasets.

    Parameters:
        results: np.ndarray
            A 2D array where rows are datasets and columns are algorithms.

    Returns:
        np.ndarray: Average ranks for each algorithm.
    """
    ranks = np.array([rankdata(-row) for row in results])
    return np.mean(ranks, axis=0)


def friedman(results):
    """
    Performs the Friedman test for differences between algorithms.

    Parameters:
        results: np.ndarray
            A 2D array where rows are datasets and columns are algorithms.

    Returns:
        tuple: Friedman statistic and p-value.
    """
    return friedmanchisquare(*results.T)


def iman_davenport(results, friedman_stat):
    """
    Performs the Iman-Davenport correction for the Friedman test.

    Parameters:
        results: np.ndarray
            A 2D array where rows are datasets and columns are algorithms.
        friedman_stat: float
            The Friedman test statistic.

    Returns:
        tuple: Iman-Davenport statistic and p-value.
    """
    n_datasets, n_algorithms = results.shape
    iman_stat = ((n_datasets - 1) * friedman_stat) / \
        (n_datasets * (n_algorithms - 1) - friedman_stat)
    df1 = n_algorithms - 1
    df2 = (n_algorithms - 1) * (n_datasets - 1)
    p_value = 1 - t.cdf(iman_stat, df2)
    return iman_stat, p_value


def bonferroni(avg_ranks, n_algorithms):
    """
    Performs the Bonferroni-Dunn post hoc test for pairwise comparisons.

    Parameters:
        avg_ranks: np.ndarray
            Average ranks of each algorithm.
        n_algorithms: int
            Number of algorithms being compared.

    Returns:
        dict: Pairwise comparisons and whether they are significant.
    """
    n_comparisons = n_algorithms * (n_algorithms - 1) // 2
    critical_value = t.ppf(1 - 0.05 / n_comparisons, n_comparisons)

    comparisons = {}
    for i in range(n_algorithms):
        for j in range(i + 1, n_algorithms):
            rank_diff = abs(avg_ranks[i] - avg_ranks[j])
            critical_diff = critical_value
            comparisons[(i, j)] = {
                "rank_difference": rank_diff,
                "significant": rank_diff > critical_diff,
            }
    return comparisons


# Example usage
if __name__ == "__main__":
    # Example: Rows are datasets, columns are algorithms
    example_results = np.array([
        [0.81, 0.83, 0.75],
        [0.78, 0.80, 0.76],
        [0.88, 0.85, 0.79],
    ])

    results = demsar(example_results)
    print("Demšar Test Results:")
    for key, value in results.items():
        print(f"{key}: {value}")

