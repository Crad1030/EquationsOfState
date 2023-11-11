import os
import numpy as np


def line_count(filename: str) -> int:
    """Counts the number of lines in a file.

    Args:
        filename: The path to the file.

    Returns:
        The number of lines in the file.
    """

    with open(filename) as f:
        return sum(1 for line in f)


def file_options(path: str, msg: str, extensions: list[str] = []) -> dict[str, int]:
    """Creates a numbered list of options to select files from, excluding files with the specified extensions.

    Args:
        path: The path to the directory containing the files.
        msg: A message to display to the user before the list of options.
        extensions: A list of file extensions to exclude from the list of options.

    Returns:
        A dictionary mapping file names to their corresponding option numbers.
    """

    print(f"\n{msg}")
    num = 0
    dictionary_of_files = {}
    os.chdir(path)
    for files in os.listdir():
        if os.path.isfile(files) and files[-4:] in extensions:
            num += 1
            dictionary_of_files[files] = num
            print(f"{num:>2d}. {files}")
    dictionary_of_files[len(dictionary_of_files)] = "Exit Program"
    print(f"{len(dictionary_of_files):>2d}. Exit Program")
    return dictionary_of_files


def key_selection(dictionary: dict[str, int], selection: int) -> str:
    """Returns the option corresponding to the selection number from file_options.

    Args:
        dictionary: A dictionary mapping file names to their corresponding option numbers.
        selection: The selection number.

    Returns:
        The name of the selected file.
    """

    selection = type_convert(selection)
    while selection > len(dictionary) or selection <= 0 or not isinstance(selection, int):
        print("Not a valid selection!")
        selection = int(input("Make a New Selection: "))

    for key, value in dictionary.items():
        if value == "Exit Program":
            exit()
        if value == selection:
            return key


def type_convert(variable: object, dtype: type = int) -> object:
    """Checks to make sure the data type is correct, if not, forces a new selection.

    Args:
        variable: The variable to check the data type of.
        dtype: The desired data type.

    Returns:
        The variable, converted to the desired data type.
    """

    while not isinstance(variable, dtype):
        try:
            variable = dtype(variable)
        except ValueError:
            print("Not a valid selection!")
            variable = input("Make a New Selection: ")

    return variable


def parse_parameters(info: list[str]) -> tuple[str, float, np.ndarray, list[str]]:
    """Parses the input parameters for the equation of state calculations.

    Args:
        info: A list of strings containing the input parameters.

    Returns:
        A tuple containing the following elements:
            * The name of the equation of state (eos).
            * The temperature (T).
            * The coefficients of the equation of state (C).
            * The units of the coefficients (units).
    """

    eos = info[0][0].upper()
    C = np.array(info[1], dtype=np.float64)
    T = float(info[2][1])
    units = " ".join(info[3]).lower().split()

    return eos, T, C, units


def pv_units(units: list[str]) -> tuple[float, float]:
    """Determines the conversion factors for pressure and volume units.

    Args:
        units: A list of strings containing the units of pressure and volume.

    Returns:
        A tuple containing the following elements:
            * The conversion factor for pressure units (press_conv).
            * The conversion factor for volume units (vol_conv).
    """

    units[0] = units[0].split("/")

    volume = {
        "dm^3": 1E-3,
        "cm^3": 1E-6,
        "l": 1E-3,
        "m^3": 1E0,
        "L": 1E-3,
    }

    if units[0][0].lower() in volume.keys():
        vol_conv = volume[units[0][0]]
    else:
        raise ValueError("Invalid volume units.")

    press = {
        "bar": 1E5,
        "torr": 133.32236842105263,
        "kilobar": 1E8,
        "mmhg": 133.322387415,
        "atm": 101325,
        "pa": 1E0,
    }

    if units[1].lower() in press.keys():
        press_conv = press[units[1].lower()]
    else:
        raise ValueError("Invalid pressure units.")

    return vol_conv, press_conv


def dimensional_shift(units: list[str], P: float, V: float, R: float) -> tuple[float, float]:
    """Converts the pressure and volume from native units to SI units.

    Args:
        units: A list of strings containing the units of pressure and volume.
        P: The pressure in native units.
        V: The volume in native units.
        R: The ideal gas constant in SI units.

    Returns:
        A tuple containing the following elements:
            * The pressure in SI units.
            * The volume in SI units.
    """

    vol_conv, press_conv = pv_units(units)
    P_SI = P * press_conv
    V_SI = V * vol_conv
    return P_SI, V_SI


def parse_input(filename: str) -> tuple[np.ndarray, np.ndarray, list]:
    """Parses the input file data.

    Args:
        filename: The path to the input file.

    Returns:
        A tuple containing the following elements:
            * A NumPy array containing the pressure data (P).
            * A NumPy array containing the volume data (V).
            * A list containing the other input parameters (info).
    """

    info = []
    P = np.array([], dtype=np.float64)
    V = np.array([], dtype=np.float64)

    with open(filename, "r") as df:
        for num, line in enumerate(df):
            line = line.strip().split()
            if num >= 5 and num != 0:
                P = np.append(P, np.float64(line[1]))
                V = np.append(V, np.float64(line[0]))
            elif num != 0 and num < 5:
                info.append(line)

    return P, V, info


def input_data(filename: str, eos: str, T: float, R: float, C: np.ndarray, lines: int) -> None:
    """Outputs the information from running the input file.

    Args:
        filename: The path to the input file.
        eos: The name of the equation of state.
        T: The temperature.
        R: The ideal gas constant.
        C: The coefficients of the equation of state.
        lines: The number of lines in the input file.
    """

    print("\n")
    print("\033[1;92mFILE INFORMATION\033[0m:")
    print(f"Using Data File       = \033[93m{filename[:-4].upper()}\033[0m")
    print(f"Equation of State     = \033[93m{eos.upper()}\033[0m")
    print(f"Lines of Data         = \033[93m{lines:d}\033[0m")
    print(f"Number of Parameters  = \033[93m{len(C):d}\033[0m")
    print(f"Initial Parameters    = \033[93m{C}\033[0m")
    print(f"Temp and Gas Constant = \033[93m{T:<10.6E} | {R:.6E}\033[0m")
    print("\n")


def center(string: str, size: int) -> None:
    """Centers the given string in a box of the specified size.

    Args:
        string: The string to center.
        size: The size of the box.
    """
    if size % 2 == 0:
        length = 21
        print(string.center((length*size)+1,"-"))
    elif size % 3 == 0:
        length = 21
        print(string.center((length*size)+1,"-"))
    elif size % 2 == 0 and size % 3 == 0:
        length = 22
        print(string.center(length*size,"-"))
    else:
        length = 22
        print(string.center(length*size,"-"))


def print_cycle(cycle: int, vlambda: float) -> None:
    """Prints the current cycle and lambda value to the console.

    Args:
        cycle: The current cycle number.
        vlambda: The current lambda value.
    """

    print("\n")
    header = f" CYCLE: {cycle} | "
    slambda = f"LAMBDA: {vlambda:.3E} "
    string = header + slambda
    print(string.center(75, "="))


def print_beta_parameters(beta: np.ndarray, old: np.ndarray, new: np.ndarray) -> None:
    """Prints the beta parameters to the console.

    Args:
        beta: The beta parameters.
        old: The previous beta parameters.
        new: The new beta parameters.
    """

    data = ""
    old, new = old.flatten(), new.flatten()
    string1, string2, string3 = " BETA ", " PREV PARAMS ", " NEW PARAMS "
    init = f"\n{string1.center(20, '-')}{string2.center(23, '-')}{string3.center(21, '-')}"
    print(init)
    for i in range(len(beta)):
        data = f"| {beta[i]:+8.11E} | {old[i]:+8.11E} | {new[i]:+8.11E} |\n"
        print(data.strip())
    print("-" * 64)


def print_structure(mat: np.ndarray, string: str) -> None:
    """Prints the given NumPy array to the console in a formatted way.

    Args:
        mat: The NumPy array to print.
        string: The string to print at the center of the output.
    """

    print("\n")
    mat_x = np.shape(mat)[0]
    mat_y = np.shape(mat)[1]
    string = " " + string + " "
    center(string, len(mat))
    for i in range(mat_x):
        for j in range(mat_y):
            if j == 0 and len(mat) != 1:
                print("|{:^+21.11E}".format(mat[i][j]), end="")
            elif j + 1 != len(mat):
                print("{:^+21.11E}".format(mat[i][j]), end="")
            elif len(mat) == 1 and j == 0:
                print("|{:^+21.11E}|".format(mat[i][j]))
            else:
                print("{:^+20.11E}|".format(mat[i][j]))
    string = ""
    center(string, len(mat))
