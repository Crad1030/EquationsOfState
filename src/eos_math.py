import numpy as np
from typing import Callable


def get_equation_of_state(eqn: str) -> Callable[[float, float, float], float]:
    """
    Returns a function that calculates the equation of state for the given gas.

    Args:
        eqn: The name of the equation of state to use, such as "van_der_waals", "redlich_kwong",
          "dieterici", "berthelot", or "virial".

    Returns:
        A function that calculates the equation of state for the given gas.
    """

    eqns = {
        "VDW": van_der_waals,
        "RK": redlich_kwong,
        "DIETERICI": dieterici,
        "BERTHELOT": berthelot,
        "VIRIAL": virial,
    }

    if eqn.upper() not in eqns:
        raise ValueError(f"Unknown equation of state: {eqn}")

    return eqns[eqn.upper()]


def van_der_waals(V: float, C: list[float], R: float = 8.31447, T: float = 273.15) -> float:
    """Calculates the Van der Waals equation of state.

    Args:
        V: The molar volume in m^3/mol.
        C: A list of two constants, C[0] and C[1].
        R: The ideal gas constant in J/mol·K.
        T: The temperature in K.

    Returns:
        The pressure in Pa.
    """

    return ((R * T) / (V - C[1])) - (C[0] / (V * V))


def redlich_kwong(V: float, C: list[float], R: float = 8.31447, T: float = 273.15) -> float:
    """Calculates the Redlich-Kwong equation of state.

    Args:
        V: The molar volume in m^3/mol.
        C: A list of two constants, C[0] and C[1].
        R: The ideal gas constant in J/mol·K.
        T: The temperature in K.

    Returns:
        The pressure in Pa.
    """

    return ((R * T) / (V - C[1])) - (C[0] / (np.sqrt(T) * V * (V + C[1])))

def berthelot(V: float, C: list[float], R: float = 8.31447, T: float = 273.15) -> float:
    """Calculates the Berthelot equation of state.

    Args:
        V: The molar volume in m^3/mol.
        C: A list of two constants, C[0] and C[1].
        R: The ideal gas constant in J/mol·K.
        T: The temperature in K.

    Returns:
        The pressure in Pa.
    """

    return ((R * T) / (V - C[1])) - (C[0] / (T * V * V))


def dieterici(V: float, C: list[float], R: float = 8.31447, T: float = 273.15) -> float:
    """Calculates the Dieterici equation of state.

    Args:
        V: The molar volume in m^3/mol.
        C: A list of two constants, C[0] and C[1].
        R: The ideal gas constant in J/mol·K.
        T: The temperature in K.

    Returns:
        The pressure in Pa.
    """

    return ((R * T) * np.exp(-C[0] / (R * T * V))) / (V - C[1])


def virial(V: float, C: list[float], R: float = 8.31447, T: float = 273.15) -> float:
    """Calculates the virial equation of state.

    Args:
        V: The molar volume in m^3/mol.
        C: A list of constants, C[0], C[1], ..., C[n-1].
        R: The ideal gas constant in J/mol·K.
        T: The temperature in K.

    Returns:
        The pressure in Pa.
    """

    total = (R * T) / V
    summ = 0.0
    for i in range(len(C)):
        summ += C[i] / (V**(i + 1))
    return total * (1 + summ)


def partial_deriv_virial(V: np.ndarray, C: np.ndarray, R: float, T: float) -> np.ndarray:
    """Calculates the partial derivatives of the virial equation of state.

    Args:
        V: A numpy array of molar volumes in m^3/mol.
        C: A numpy array of constants.
        R: The ideal gas constant in J/mol·K.
        T: The temperature in K.

    Returns:
        A numpy array of partial derivatives, with shape (len(V), len(C)).
    """

    derivs = np.zeros((len(V), len(C)))
    for i in range(len(V)):
        for j in range(len(C)):
            derivs[i][j] = (R * T) / V[i]**(j + 2)
    return derivs


def error(func: object, obs: np.ndarray, V: np.ndarray, C: np.ndarray, R: float = 8.31447, T: float = 273.15) -> np.ndarray:
    """Calculates the error between calculated and experimental values.

    Args:
        func: A function that calculates the pressure given the molar volume and constants.
        obs: A numpy array of experimental pressures in Pa.
        V: A numpy array of molar volumes in m^3/mol.
        C: A numpy array of constants.
        R: The ideal gas constant in J/mol·K.
        T: The temperature in K.

    Returns:
        A numpy array of errors, with shape (len(V),).
    """

    return obs - func(V, C, R, T)


def regression(obs: np.ndarray, mean: float) -> float:
    """Calculates the regression coefficient of a set of data points.

    Args:
        obs: A numpy array of data points.
        mean: The mean of the data points.

    Returns:
        The regression coefficient, a value between 0 and 1, with a higher value indicating a better fit.
    """

    total = 0.0
    for i in range(len(obs)):
        total += (obs[i] - mean) ** 2
    return 1 - total / np.var(obs)


def jacobian_matrix_second(func: object, C: np.ndarray, args: list = [], h: float = 1E-6) -> np.ndarray:
    """Second-order finite difference approximation of the Jacobian matrix for equations of state.

    Args:
        func: A function that calculates the pressure given the molar volume and constants.
        C: A numpy array of constants.
        args: A list of additional arguments to pass to the function.
        h: The step size for the finite difference approximation.

    Returns:
        A numpy array of the Jacobian matrix, with shape (len(args[2]), len(C)).
    """

    pderiv = np.zeros((len(args[2]), len(C)), dtype=np.float64)
    for j in range(len(C)):
        C[j] += (2 * h)
        df1 = func(args[2], C, R=args[0], T=args[1])
        C[j] -= h
        df2 = func(args[2], C, R=args[0], T=args[1])
        C[j] -= (2 * h)
        df3 = func(args[2], C, R=args[0], T=args[1])
        C[j] -= h
        df4 = func(args[2], C, R=args[0], T=args[1])
        C[j] += (2 * h)

        for i in range(len(args[2])):
            pderiv[i][j] = ((-df1[i] + (8 * df2[i]) - (8 * df3[i]) + df4[i])) / (12 * h)

    return pderiv
