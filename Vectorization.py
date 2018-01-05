import pandas as pd
import math

# data = [[2, 3, 4, 6],[2, 3, 4, 5, 6],[2, -2, 4, -6]]

def changetovector(data):

    result = []
    for datastr in data:

        series = pd.Series(datastr)
        t = series.describe()
    #t0: count t1 mean t2 std t3 min t5 median t7 max t4 25% t6 75%
        datalist = [t[0],t[1],t[2],t[3],t[5],t[7],t[4],t[6]]
    #add skew
        skew = series.skew()
        datalist.append(skew)
    #add kurtosis
        kurtosis = series.kurtosis()
        datalist.append(kurtosis)
        result.extend(datalist)
    for i in range(len(result)):
        if math.isnan(result[i]):
            result[i] = 0

    return result

# res = changetovector(data)

# print(res)