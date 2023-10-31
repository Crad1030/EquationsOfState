import os, sys
import numpy as np

def EOS(eqn):
    """ Picks the equation of state and returns the function """
    eqns = {"VDW":VDW,"RK":RK,"DIETERICI":DIETERICI,"BERTHELOT":BERTHELOT,"VIRIAL":VIRIAL}
    for keys, values in eqns.items():
        if keys == eqn.upper():
            func = values
    return func

def VDW(V, C, R=8.31447, T=273.15):
    return ((R * T) / (V - C[1])) - (C[0] / (V * V))

def RK(V, C, R=8.31447, T=273.15):
    return ((R * T) / (V - C[1])) - (C[0] / (np.sqrt(T) * V * (V + C[1])))

def BERTHELOT(V, C, R=8.31447, T=273.15):
    return ((R * T) / (V - C[1])) - (C[0] / (T * V * V))

def DIETERICI(V, C, R=8.31447, T=273.15):
    return ((R * T)*np.exp(-C[0]/(R * T * V))) / (V - C[1])

def VIRIAL(V, C, R=8.31447, T=273.15):
    total = (R * T) / V
    summ = 0.0
    for i in range(len(C)):
        summ += C[i]/(V**(i+1))
    return total * (1 + summ)

def partial_deriv_virial(V, C, R, T):
    derivs = np.zeros((len(V), len(C)))
    for i in range(len(V)):
        for j in range(len(C)):
            derivs[i][j] = (R * T) / V[i]**(j+2)
    return derivs

def error(func,obs,V,C,R=8.31447,T=273.15):
    """ Returns the error of calculated values vs experimental """
    return np.float64(obs - func(V,C,R=R,T=T))

def regression(obs, mean):
    """ Regression using data and mean of data """
    total = 0
    for i in range(len(obs)):
        total += obs[i] - mean
    return 1 - abs(total)

def jacobian_matrix_first(func, C, args=[], h=1E-6):
    """ First-order finite difference approx. Jacobian for equations of state """
    pderiv = np.zeros([len(args[2]),len(C)],dtype=np.float64)
    for j in range(len(C)):
        C[j]         = C[j] + (2*h)
        df1          = func(args[2],C,R=args[0],T=args[1])
        C[j]         = C[j] - (4*h)
        df2          = func(args[2],C,R=args[0],T=args[1])
        for i in range(len(args[2])):
            pderiv[i][j] = (df1[j] - df2[j]) / (2*h)
        C[j]         = C[j] + (2*h)
    return pderiv

def jacobian_matrix_second(func, C, args=[], h=1E-6):
    """ Second-order finite difference approx. Jacobian for equations of state """
    pderiv = np.zeros([len(args[2]),len(C)],dtype=np.float64)
    for j in range(len(C)):
        C[j]         = C[j] + (2*h)
        df1          = func(args[2],C,R=args[0],T=args[1])
        C[j]         = C[j] - h
        df2          = func(args[2],C,R=args[0],T=args[1])
        C[j]         = C[j] - (2*h)
        df3          = func(args[2],C,R=args[0],T=args[1])
        C[j]         = C[j] - h
        df4          = func(args[2],C,R=args[0],T=args[1])
        for i in range(len(args[2])):
            pderiv[i][j] = ((-df1[i] + (8*df2[i]) - (8*df3[i]) + df4[i])) / (12*h)
        C[j]         = C[j] + (2*h)
    return pderiv

def jacobian_matrix_third(func, C, args=[], h=1E-8):
    """ Third-order finite difference approx. Jacobian for equations of state """
    pderiv = np.zeros([len(args[2]),len(C)],dtype=np.float64)
    for j in range(len(C)):
        C[j]         = C[j] + (3*h)
        df1          = func(args[2],C,R=args[0],T=args[1])
        C[j]         = C[j] - h
        df2          = func(args[2],C,R=args[0],T=args[1])
        C[j]         = C[j] - h
        df3          = func(args[2],C,R=args[0],T=args[1])
        C[j]         = C[j] - (2*h)
        df4          = func(args[2],C,R=args[0],T=args[1])
        C[j]         = C[j] - h
        df5          = func(args[2],C,R=args[0],T=args[1])
        C[j]         = C[j] - h
        df6          = func(args[2],C,R=args[0],T=args[1])
        for i in range(len(args[2])):
            pderiv[i][j] = ((df1[i] - (9*df2[i]) + (45*df3[i]) - (45*df4[i]) + (9*df5[i]) - df6[i]) / (60*h))
        C[j]         = C[j] + (3*h)
    return pderiv

def jacobian_matrix_fourth(func, C, args=[], h=1E-7):
    """ Fourth-order finite difference approx. Jacobian for equations of state """
    pderiv = np.zeros([len(args[2]),len(C)],dtype=np.float64)
    for j in range(len(C)):
        C[j]         = C[j] + (4*h)
        df1          = func(args[2],C,R=args[0],T=args[1])
        C[j]         = C[j] - h
        df2          = func(args[2],C,R=args[0],T=args[1])
        C[j]         = C[j] - h
        df3          = func(args[2],C,R=args[0],T=args[1])
        C[j]         = C[j] - h
        df4          = func(args[2],C,R=args[0],T=args[1])
        C[j]         = C[j] - (2*h)
        df5          = func(args[2],C,R=args[0],T=args[1])
        C[j]         = C[j] - h
        df6          = func(args[2],C,R=args[0],T=args[1])
        C[j]         = C[j] - h
        df7          = func(args[2],C,R=args[0],T=args[1])
        C[j]         = C[j] - h
        df8          = func(args[2],C,R=args[0],T=args[1])
        for i in range(len(args[2])):
            pderiv[i][j] = (((-3*df1[i])+(32*df2[i])-(168*df3[i])+(672*df4[i])-(672*df5[i])+(168*df6[i])-(32*df7[i])+(3*df8[i]))/(840*h))
        C[j]         = C[j] + (4*h)
    return pderiv