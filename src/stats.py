
def chi_squared(error):
    summ = 0.0
    for i in error:
        summ += i**2
    return summ

def sigma_squared(error, Pobs, C):
    summ = 0.0
    for i in error:
        summ += i**2
    return summ / (len(Pobs) - len(C))

def mean(Pobs):
    return sum(Pobs)/len(Pobs)

def total_sum_of_squares(Pobs, mean_pobs):
    summ = 0.0
    for i in Pobs:
        summ += (i - mean_pobs)**2
    return summ