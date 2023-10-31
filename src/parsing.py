import os
import numpy as np


def line_count(filename):
    """ Returns an integer (number of lines in the file) """
    with open(filename) as fn:
        for num, line in enumerate(fn):
            pass
        return num + 1

def file_options(path, msg, extensions=[]):
    """ Creates a numbered list of options to select files from and excludes files with 'extensions' """
    print("\n{} ".format(msg))
    num = 0
    dictionary_of_files = {}
    os.chdir(path)
    for files in os.listdir():
        if os.path.isfile(files) == True and files[-4:] in extensions:
            num += 1
            dictionary_of_files[files] = num
            print("{:>2d}. {}".format(num,files))
    dictionary_of_files[len(dictionary_of_files)] = "Exit Program"
    print("{:>2d}. {}".format(len(dictionary_of_files),"Exit Program"))
    return dictionary_of_files

def key_selection(dictionary, selection):
    """ Returns the option corresponding to the selection number from file_options """
    selection = type_convert(selection)
    while selection > len(dictionary) or selection <= 0 or type(selection) != int:
        print("Not a valid selection!")
        selection = int(input("Make a New Selection: "))
    for key, value in dictionary.items():
        if value == "Exit Program": 
            exit()
        if value == selection:
            return key

def type_convert(variable, dtype=int):
    """ Checks to make sure data type is correct, if not, forces new selection """
    while type(variable) != dtype:
        try:
            variable = dtype(variable)
        except ValueError:
            print("Not a valid selection!")
            variable = input("Make a New Selection: ")
    return variable

def parse_parameters(info):
    """ Fixing data types and defining parameters """
    eos   = info[0][0].upper()
    C     = np.array(info[1],dtype=np.float64)
    T     = float(info[2][1])
    units = " ".join(info[3]).lower().split()
    return eos,T,C,units

def gas_constant(units):
    """ Determines the R constant based on the units of Pressure """
    # R = {"atm"    :8.20574E-2, "pa"  :8.31447, "kilobar":8.31447E1,
    #      "mmhg"   :62.364,     "torr":62.364,  "bar"    :8.31447E-2}
    # for key,value in R.items():
    #     if units[1].lower() == key:
    return 8.31446261815324

def pv_units(units):
    """ Determines pressure and volume """
    units[0] = units[0].split("/")
    volume   = {"dm^3":1E-3, "cm^3":1E-6, 
                "l"   :1E-3, "m^3" :1E0, "L" :1E-3}
    if units[0][0].lower() in volume.keys():
        vol_conv = volume[units[0][0]]
    press = {"bar" :1E5,     "torr":133.32236842105263, "kilobar":1E8, 
             "mmhg":133.322387415, "atm" :101325,  "pa"     :1E0}
    if units[1].lower() in press.keys():
        press_conv = press[units[1]]
    return vol_conv, press_conv

def dimensional_shift(units, P, V, R):
    """ Converts native units to SI units """
    vconv, pconv = pv_units(units)
    return P*pconv, V*vconv

def parse_input(filename):
    """ Coarse parsing of input file data """
    info = []
    P    = np.array([],dtype=np.float64)
    V    = np.array([],dtype=np.float64)
    with open(filename,"r") as df:
        for num, line in enumerate(df):
            line = line.strip().split()
            if num >= 5 and num != 0:
                P = np.append(P,np.float64(line[1]))
                V = np.append(V,np.float64(line[0]))
            elif num != 0 and num < 5:
                info.append(line)
    return P, V, info

def input_data(filename,eos,T,R,C,lines):
    """ Outputs the information from running the input file """
    print("\n".strip())
    print("\033[1;92mFILE INFORMATION\033[0m:")
    print("Using Data File       = \033[93m{:15s}\033[0m".format(filename[:-4].upper()))
    print("Equation of State     = \033[93m{:15s}\033[0m".format(eos.upper()))
    print("Lines of Data         = \033[93m{:d}\033[0m".format(lines))
    print("Number of Parameters  = \033[93m{:d}\033[0m".format(len(C)))
    print("Initial Parameters    = \033[93m{}\033[0m".format(C))
    print("Temp and Gas Constant = \033[93m{:<10.6E} | {:.6E}\033[0m".format(T,R))
    print("\n".strip())

def center(string,size):
    """ Print formatting for the -w command line flag """
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

def print_cycle(cycle,vlambda):
    """ Print formatting for the -w command line flag """
    print("\n".strip())
    header  = " CYCLE: " + str(cycle) + "  | "
    slambda = "LAMBDA: " + "{:.3E}".format(vlambda) + " "
    string  = header + slambda
    print(string.center(75,"="))

def print_beta_parameters(beta,old,new):
    """ Print formatting for the -w command line flag """
    data = ""
    old, new = old.flatten(), new.flatten()
    string1,string2,string3 = " BETA " , " PREV PARAMS " , " NEW PARAMS "
    init = "\n"+string1.center(20,"-") + string2.center(23,"-") + string3.center(21,"-")
    print(init)
    for i in range(len(beta)):
        data = "| {:+8.11E} | {:+8.11E} | {:+8.11E} |\n".format(beta[i],old[i],new[i])
        print(data.strip())
    print("-"*64)

def print_structure(mat,string):
    """ Print formatting for the -w command line flag """
    print("\n".strip())
    mat_x = np.shape(mat)[0]
    mat_y = np.shape(mat)[1]
    string = " " + string + " "
    center(string,len(mat))
    for i in range(mat_x):
        for j in range(mat_y):
            if j == 0 and len(mat) != 1:
                print("|{:^+21.11E}".format(mat[i][j]),end="")
            elif j+1 != len(mat):
                print("{:^+21.11E}".format(mat[i][j]),end="")
            elif len(mat) == 1 and j == 0:
                print("|{:^+21.11E}|".format(mat[i][j]))
            else:
                print("{:^+20.11E}|".format(mat[i][j]))
    string = ""
    center(string,len(mat))