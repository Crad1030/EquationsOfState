from eos_math import *

def chi_squared(error: np.ndarray) -> float:
    """Calculates the chi-squared value for the given error array.

    Args:
        error: A NumPy array containing the errors.

    Returns:
        The chi-squared value.
    """

    summ = 0.0
    for i in error:
        summ += i ** 2
    return summ


def sigma_squared(error: np.ndarray, Pobs: np.ndarray, C: np.ndarray) -> float:
    """Calculates the squared standard deviation of the error.

    Args:
        error: A NumPy array containing the errors.
        Pobs: A NumPy array containing the observed pressures.
        C: A NumPy array containing the coefficients of the equation of state.

    Returns:
        The squared standard deviation of the error.
    """

    summ = 0.0
    for i in error:
        summ += i ** 2
    return summ / (len(Pobs) - len(C))


def mean(Pobs: np.ndarray) -> float:
    """Calculates the mean of the given pressure array.

    Args:
        Pobs: A NumPy array containing the observed pressures.

    Returns:
        The mean of the pressure array.
    """

    return sum(Pobs) / len(Pobs)


def total_sum_of_squares(Pobs: np.ndarray, mean_pobs: float) -> float:
    """Calculates the total sum of squares for the given pressure array and mean.

    Args:
        Pobs: A NumPy array containing the observed pressures.
        mean_pobs: The mean of the pressure array.

    Returns:
        The total sum of squares.
    """

    summ = 0.0
    for i in Pobs:
        summ += (i - mean_pobs) ** 2
    return summ