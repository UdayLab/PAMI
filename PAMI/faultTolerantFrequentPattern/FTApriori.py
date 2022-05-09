import sys
import time
import resource
import math
import itertools
path=sys.argv[1]
output = sys.argv[2]
minSup = int(sys.argv[3])
itemSup = int(sys.argv[4])
minLength = int(sys.argv[5])
faultTolerance = int(sys.argv[6])
mapSupport = {}
finalPatterns = {}
Database = []

def Count(k):
    count = 0
    items = []
    k = list(k)
    n = len(k)-faultTolerance
    c = itertools.combinations(k, n)
    count = 0
    for j in c: 
        j = list(j)
        for i in Database:
            if set(j).issubset(i):
                count += 1
                items.append(i)
    items =  list(set(map(tuple,items)))
    print(k, items, count)
    return len(items), items
            
    
def countItemSupport(itemset, transactions):
    tids = {}
    res = True
    for i in itemset:
        for k in transactions:
            if i in k:
                if i not in tids:
                    tids[i] = 1
                else:
                    tids[i] += 1
    #print(itemset, transactions, tids)
    for x, y in tids.items():
        if y < itemSup:
            res = False
    return res

def getFaultPatterns():
    
    l = [k for k,v in mapSupport.items()]
    for i in range(2, len(l)+1):
        c = itertools.combinations(l,i)
        for j in c:
            support, items = Count(j)
            res = countItemSupport(j, items)
            if len(j)> minLength and len(j)>=faultTolerance and support >= minSup and res == True:
                finalPatterns[tuple(j)] = support
                    

with open(path, 'r') as f:
    for line in f:
        li = line.split()
        Database.append(li)
        for i in li:
            if i not in mapSupport:
                mapSupport[i] = 1
            else:
                mapSupport[i] +=1
l = [k for k, v in mapSupport.items()]
for i in range(len(l)):
    for j in range(i+1, len(l)):
        x, y = l[i], l[j]
        li = [x, y]
        count = 0
        tids = {x:0, y:0}
        for k in Database:
            if x in k and y in k:
                count += 1
                tids[x] += 1
                tids[y] += 1
            if x in k and y not in k:
                count += 1
                tids[x] = 1
            if x not in k and y in k:
                count += 1
                tids[y] += 1
        re = True
        for x, y in tids.items():
            if y< itemSup:
                re = False
        finalPatterns[tuple(li)] = count

getFaultPatterns() 

print("...")
with open(output, 'w+') as f:
    for x,y in finalPatterns.items():
        s = str(x) + ":" + str(y)
        f.write("%s \n" % s)
for x,y in finalPatterns.items():
    print(x,y)
