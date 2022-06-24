import pandas as pd
import numpy as np

import sys

sys.stdout = open("iris_summary.txt", "w")

iris = pd.read_csv("irisdata.txt", header=None)

flavors = list(set(iris[4]))

flavors_data = {}

for flav in flavors:
    tmp = (iris[4] == flav).astype(np.int)
    par0 = tmp * iris[0]
    par1 = tmp * iris[1]
    par2 = tmp * iris[2]
    par3 = tmp * iris[3]
    tmp2 = []
    for i in range(len(tmp)):
        if (tmp[i] == 1):
            tmp2.append([par0[i], par1[i], par2[i], par3[i]])
    flavors_data[flav] = np.array(tmp2)

iris = iris.values

names = ["sepal length", "sepal width", "petal length", "petal width"]
maxlen = max([len(i) for i in names])

header = " " * maxlen + '\t' + "\t".join(flavors) + '\tTotal'
print(header)
print("=" * len(header) + "=" * (5 * header.count("\t")))


def getStatL(arr):  # arr=flavors_data[flavors[0]][:,0]
    res = {}
    res['N'] = len(arr)
    res['mean'] = arr.mean()
    res['min'] = min(arr)
    res['max'] = max(arr)
    tmp = arr.copy()
    tmp.sort()
    res['median'] = (tmp[int(len(tmp) / 2)] + tmp[int(len(tmp) / 2) + 1]) / 2
    res['standard deviation'] = ((arr - arr.mean()) ** 2).mean() ** 0.5
    res['< 5'] = sum((arr < 5).astype(np.int))
    res['< 5%'] = res['< 5'] / len(arr) * 100
    res['>=5 and <6'] = sum((arr >= 5).astype(np.int) * (arr < 6).astype(np.int))
    res['>=5 and <6%'] = res['>=5 and <6'] / len(arr) * 100
    res['>=6 and <7'] = sum((arr >= 6).astype(np.int) * (arr < 7).astype(np.int))
    res['>=6 and <7%'] = res['>=6 and <7'] / len(arr) * 100
    res['>=7'] = sum((arr >= 7).astype(np.int))
    res['>=7%'] = res['>=7'] / len(arr) * 100
    return res


def getStatW(arr):  # arr=flavors_data[flavors[0]][:,0]
    res = {}
    res['N'] = len(arr)
    res['mean'] = arr.mean()
    res['min'] = min(arr)
    res['max'] = max(arr)
    tmp = arr.copy()
    tmp.sort()
    res['median'] = (tmp[int(len(tmp) / 2)] + tmp[int(len(tmp) / 2) + 1]) / 2
    res['standard deviation'] = ((arr - arr.mean()) ** 2).mean() ** 0.5
    res['< 3'] = sum((arr < 3).astype(np.int))
    res['< 3%'] = res['< 3'] / len(arr) * 100
    res['>=3 and <3.5'] = sum((arr >= 3).astype(np.int) * (arr < 3.5).astype(np.int))
    res['>=3 and <3.5%'] = res['>=3 and <3.5'] / len(arr) * 100
    res['>=3.5 and <4'] = sum((arr >= 3.5).astype(np.int) * (arr < 4).astype(np.int))
    res['>=3.5 and <4%'] = res['>=3.5 and <4'] / len(arr) * 100
    res['>=4'] = sum((arr >= 4).astype(np.int))
    res['>=4%'] = res['>=4'] / len(arr) * 100
    return res


def buildStatLine(f, stat):
    tmp = f.upper() + " " * max(maxlen - len(f), 0) + '\t'
    for flav in flavors:
        tmp2 = str(round(flavors_stat[flav][f], 1))
        if (f + "%" in flavors_stat[flav]):
            tmp2 += "(" + str(round(flavors_stat[flav][f + "%"], 1)) + "%)"
        tmp += tmp2
        if (len(tmp2) > 7):
            tmp += '\t'
        else:
            tmp += '\t\t'
    tmp += str(round(flavors_stat['total'][f], 1))
    if (f + "%" in flavors_stat['total']): tmp += "(" + str(round(flavors_stat['total'][f + "%"], 1)) + "%)"
    return tmp


for name in names:
    print(name.upper())
    flavors_stat = {}
    getStat = getStatL
    if ("width" in name): getStat = getStatW
    for flav in flavors:
        flavors_stat[flav] = getStat(flavors_data[flav][:, names.index(name)])
    flavors_stat["total"] = getStat(iris[:, names.index(name)])
    stat_features = list(flavors_stat["total"].keys())
    for f in stat_features:
        if ("%" in f): continue
        print(buildStatLine(f, flavors_stat))
    print("=" * len(header) + "=" * (5 * header.count("\t")))
