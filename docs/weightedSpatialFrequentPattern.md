# Mining Spatial(Neighbourhood) Weighted Frequent Patterns in Spatiotemporal Databases

## 1. What is spatial weighted frequent pattern mining?

weighted Frequent neighbourhood pattern mining aims to discover all interesting patterns in a transactional database that have **weighted sum** no less than the user-specified **minimum weighted sum** (**minWS**) constraint, and **dist** no greater than the user-specified **maximum distance (_maxDist_)**.  The **minWS** controls the minimum number of transactions that a pattern must appear in a database. The **minWeight** controls the minimum weight of item. The **maxDist** controls the maximum distance between two items.<br>

Reference: R. U. Kiran, P. P. C. Reddy, K. Zettsu, M. Toyoda, M. Kitsuregawa and P. K. Reddy, "Efficient Discovery of Weighted Frequent Neighborhood Itemsets in Very Large Spatiotemporal Databases," in IEEE Access, vol. 8, pp. 27584-27596, 2020, doi: 10.1109/ACCESS.2020.2970181.

## 2. What is the transactional weighted database?

A transactional database is a collection of transactions, where each transaction contains a transaction-identifier and a set of items and its respective weights. <br> A hypothetical transactional database containing the items **_a, b, c, d, e, f, and g_** as shown below

|tid| Transactions| Weights |
| --- | --- | --- |
| 1 | a b f g | 20 15 20 20 |
| 2 | a c f g | 5 30 20 10 |
| 3 | d f g | 30 20 15|
| 4 | b c d | 60 80 10 |
| 5 | b c d e | 60 40 20 5
| 6 | a b c e g | 10 20 45 10 25 |

__Note:__  Duplicate items must not exist in a transaction.

## 3. What is acceptable format of a transactional database with respective weights in PAMI

Each row in a transactional database must contain only items. The frequent pattern mining algorithms in PAMI implicitly assume the row number of a transaction as its transactional-identifier to reduce storage and processing costs. A sample transactional database, say sampleInputFile.txt, is provided below.

a b f g:20 15 20 20 <br>
a c f g:5 30 20 10 <br>
d f g:30 20 15 <br>
b c d:60 80 10 <br>
b c d e:60 40 20 5 <br>
a b c e g:10 20 45 10 25  <br>

## 4. What is a neighborhood database? 

A neighborhood database contains items and their neighbors. An item *x* is said to be a neighbor of *y* if the distance between *x* and *y* is no more than the user-specified *maximum distance* threshold value.<br>

A hypothetical neighborhood database containing the items **_a, b, c, d, e, f and g_** is shown below.

| Items | neighbours |
| --- | --- |
| a | b, c, e |
| b | a, d, e |
| c | a, d, e | 
| d | b, c, e, f |
| e | a, b, c, d |
| f | d, g |
| g | f |

The methodology to create a neighborhood database file from a given geo-referenced database has been described in the manual [creatingNeighborhoodFile.pdf](creatingNeighborhoodFile.pdf)

## 5. What is the acceptable format of a spatial database?

The format of the neighborhood database is similar to that of a transactional database. That is, each transaction must contain a set of items. In a transaction, the first item represents the key item, while the remaining items represent the neighbors of the first item. 

A sample neighborhood file, say [sampleNeighbourFile.txt](sampleNeighbourFile.txt), is provided below:

a b c e <br>
b a d e <br>
c a d e <br>
d b c e f <br>
e a b c d <br>
f d g <br>
g f <br>

## 6. What is the need of understanding the statisctics of transaction weighted database?

To understand about the database. The below code will give the detail about the transactional database.
* Total number of transactions (Database size)
* Total number of unique items in database
* Minimum lenth of transaction that existed in database
* Average length of all transactions that exists in database
* Maximum length of transaction that existed in database
* Standard deviation of transaction length
* Variance in transaction length
* Sparsity of database

The below sample code prints the statistical details of a database.

```python 
import PAMI.extras.dbStats.transactionalDatabaseStats as stats 
obj = stats.transactionalDatabaseStats('sampleInputFile.txt', ' ') 
obj.run() 
obj.printStats() 
```

## 7. What are the input parameters to a spatial weighted frequent pattern mining algorithm?

Algorithms to mine the spatial weighted frequent patterns require transactional database with weights, neighborhood database, a user-specified *minSup* constraint and a Algorithms to mine the weighted frequent spatial patterns require transactional database with weights, neighborhood database, a user-specified *minWeight* constraint. Please note that *maxDist* constraint has been used in prior to create a neighborhood database file. 

. Please note that *maxDist* constraint has been used in prior to create a neighborhood database file. 

* __Transactional database__  <br> Acceptable formats:
> * String : E.g., 'transactionalDatabase.txt'
> * URL  : E.g., https://u-aizu.ac.jp/~udayrage/datasets/transactionalDatabases/transactional_T10I4D100K.csv
> * DataFrame with the header titled 'Transactions'

* __Neighbourhood database__  <br> Acceptable formats:
> * String : E.g., 'NeighbourhoodDatabase.txt'
> * URL  : E.g., https://u-aizu.ac.jp/~udayrage/datasets/transactionalDatabases/transactional_T10I4D100K.csv
> * DataFrame with the header titled 'item' and 'Neighbours'

* __Weight database__  <br> Acceptable formats:
> * String : E.g., 'WeightDatabase.txt'
> * URL  : E.g., https://u-aizu.ac.jp/~udayrage/datasets/transactionalDatabases/transactional_T10I4D100K.csv
> * DataFrame with the header titled 'items' and 'weights'

* __minSup__  <br> specified in 
> * __count (beween 0 to length of a database)__ or 
> * [0, 1]

* __minWeight__  <br> specified in 
> * __count (beween 0 to length of a database)__ or 
> * [0, 1]

* __seperator__ <br> default seperator is '\t' (tab space)

## 8. How to store the output of a weighted frequent neighbourhood pattern mining algorithm?
The patterns discovered by a weighted frequent neighbourhood pattern mining algorithm can be saved into a file or a data frame.

## 9. How to execute a weighted frequent neighbourhood pattern mining algorithms in a terminal?

* Download the PAMI source code from github.
* Unzip the PAMI source code folder and enter into weighted frequent neighbourhood pattern folder.
* Enter into weightedFrequentNeighbourhoodPattern folder
* Enter into a specific folder and execute the  following command on terminal.

__syntax:__ python3 algorithmName.py `<path to the input file>` `<path to the output file>` `<path to the neighbourhood file>` `<path to the weight file>` `<minSup>` `<minWeight>` `<seperator>`

## Sample command to execute the WFIM code in weightedFrequentPattern folder

__Example:__ python3 `WFIM.py` `inputFile.txt` `outputFile.txt` `neighbourSample.txt` `weightSample.txt` `3` `2` `5` `' '`

## How to execute a weighted frequent pattern neighbourhood mining algorithm in a Jupyter Notebook?

- Install the PAMI package from the PYPI repository by executing the following command:   **pip3 install PAMI**
* Run the below sample code by making necessary changes


```python
import PAMI.weightedFrequentNeighbourhoodPattern.basic.SWFPGrowth as alg 


iFile = 'SWFPWeightSample.txt'     #specify the input transactional database 
nFile = 'SWFPNeighbourSample.txt'  #specify the input neighbourhood database 
minWS = 150  #specify the minSupvalue  
seperator = ' ' #specify the seperator. Default seperator is tab space. 
oFile = 'weightedSpatialPatterns.txt'   #specify the output file name

obj = alg.SWFPGrowth(iFile, nFile, minWS, seperator) #initialize the algorithm 
obj.startMine()                       #start the mining process 
obj.save(oFile)               #store the patterns in file
df = obj.getPatternsAsDataFrame()     #Get the patterns discovered into a dataframe 
obj.printResults()                      #Print the statistics of mining process
```

    Weighted Frequent patterns were generated successfully using SWFPGrowth algorithm
    Total number of  Weighted Spatial Frequent Patterns: 8
    Total Memory in USS: 114229248
    Total Memory in RSS 153247744
    Total ExecutionTime in ms: 0.0018541812896728516


The weightedPatterns.txt file contains the following patterns (*format:* pattern:support): !cat weightedPatterns.txt


```python
!cat weightedPatterns.txt
```

    cat: weightedPatterns.txt: No such file or directory


The dataframe containing the patterns is shown below:


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Patterns</th>
      <th>Support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>170</td>
    </tr>
    <tr>
      <th>1</th>
      <td>e</td>
      <td>225</td>
    </tr>
    <tr>
      <th>2</th>
      <td>e b</td>
      <td>215</td>
    </tr>
    <tr>
      <th>3</th>
      <td>b</td>
      <td>245</td>
    </tr>
    <tr>
      <th>4</th>
      <td>b d</td>
      <td>270</td>
    </tr>
    <tr>
      <th>5</th>
      <td>c</td>
      <td>270</td>
    </tr>
    <tr>
      <th>6</th>
      <td>c d</td>
      <td>150</td>
    </tr>
    <tr>
      <th>7</th>
      <td>d</td>
      <td>325</td>
    </tr>
  </tbody>
</table>
</div>


