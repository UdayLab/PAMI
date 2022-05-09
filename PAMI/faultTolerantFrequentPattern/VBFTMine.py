import sys
import time
import resource
import math
import numpy as np
path=sys.argv[1]
output=sys.argv[2]
itemSup=int(sys.argv[3])
minSup=int(sys.argv[4])
faultTolerance=int(sys.argv[5])
maxPer=int(sys.argv[6])
Vector={}
lno=0
plist=[]
transactions=[]
items=[]
def Per_Sup(tids):
    per=0
    cur=0
    sup=0
    for i in range(len(tids)):
        if tids[i]==1:
            per=max(per,cur-i)
            if per>maxPer:
                return [0,0]
            cur=i
            sup+=1
    per=max(per,lno-cur)
    return [sup,per]
def Count(tids):
    count=0
    for i in tids:
        if i==1:
            count+=1
    return count
         
with open(path,'r') as f:
    for line in f:
        lno+=1
        l=line.split()
        transactions.append(l)
        for i in l[1:]:
            if i not in plist:
                plist.append(i)
for i in transactions:
    for j in plist:
        count=0
        if j in i:
            count=1
        if j in Vector:
            Vector[j].append(count)
        else:
            Vector[j]=[count]
for x,y in Vector.items():
    v=Count(y)
    if v>=itemSup:
        items.append(x)
        
def save(prefix,suffix,tidsetx):
        if(prefix==None):
            prefix=suffix
        else:
            prefix=prefix+suffix
        prefix=list(set(prefix))
        prefix.sort()
        val=Count(tidsetx)
        print(prefix,val)
        
def processEquivalenceClass(prefix,itemsets,tidsets):
        if(len(itemsets)==1):
            i=itemsets[0]
            tidi=tidsets[0]
            save(prefix,[i],tidi)
            return
        for i in range(len(itemsets)):
            itemx=itemsets[i]
            if(itemx==None):
                continue
            tidsetx=tidsets[i]
            classItemsets=[]
            classtidsets=[]
            itemsetx=[itemx]
            for j in range(i+1,len(itemsets)):
                itemj=itemsets[j]
                tidsetj=tidsets[j]
                y=list(np.array(tidsetx) & np.array(tidsetj))
                total=Count(y)
                if total>=itemSup:
                    classItemsets.append(itemj)
                    classtidsets.append(y)
            if(len(classItemsets)>0):
                newprefix=list(set(itemsetx))+prefix
                processEquivalenceClass(newprefix, classItemsets,classtidsets,classItemSets)
            save(prefix,list(set(itemsetx)),itemTidsi)

def Mine(plist):
    for i in range(len(plist)):
            itemx=plist[i]
            tidsetx=Vector[itemx]
            itemsetx=[itemx]
            itemsets=[]
            itemSets=[]
            tidsets=[]
            for j in range(i+1,len(plist)):
                itemj=plist[j]
                tidsetj=Vector[itemj]
                y1=list(np.array(tidsetx) & np.array(tidsetj))
                total=Count(y1)
                if total>=itemSup:
                    itemsets.append(itemj)
                    tidsets.append(y1)
            if(len(itemsets)>0):
                processEquivalenceClass(itemsetx,itemsets,tidsets)
            save(None,itemsetx,tidsetx)
    
               
Mine(items)
