# Mining High-Utility Patterns in Utility Databases

## 1. What is High-Utility pattern mining?

Frequent pattern mining (FPM) is an important knowledge discovery technique with many real-world applications. It involves finding all frequently occurring patterns in a (binary) transactional database. A fundamental limitation of this technique is **its inability to find interesting patterns in a quantitative transactional database.** High utility pattern mining (HUPM) has been introduced to tackle this limitation.


HUPM generalizes the model of frequent patterns by discovering all patterns that have *utility* more than the user-specified **minimum utility** (*minUtil*) in a quantitative transactional database. The *minUtil* constraint controls the minimum utility (or value) of a pattern must maintain in a database. 

Reference: Hong Yao and Howard J. Hamilton. 2006. Mining itemset utilities from transaction databases. Data Knowl. Eng. 59, 3 (December 2006), 603–626. [Link](https://doi.org/10.1016/j.datak.2005.10.004)

## 2. What is a utility database?

A utility database consists of an 'internal utility database' and an  'external utility database'.

In an internal utility database,  every transaction contains a set of items and a positive integer called **_internal utility_** respectively.

In an external utility database, every transaction contains an item and it external utility value. 


An hypothetical internal utility database is shown in below table.

| Transactions                  |
|-------------------------------|
| (a,2) (b,3) (c,1) (g,1)       |                         
| (b,3) (c,2) (d,3) (e,2)       |                         
| (a,2) (b,1) (c,3) (d,4)       |                          
| (a,3) (c,2) (d,1) (f,2)       |                          
| (a,3) (b,1) (c,2) (d,1) (g,2) |                    
| (c,2) (d,2) (e,3) (f,1)       |                          
| (a,2) (b,1) (c,1) (d,2)       |                          
| (a,1) (e,2) (f,2)             |
| (a,2) (b,2) (c,4) (d,2)       |
| (b,3) (c,2) (d,2) (e,2)       |


A hypothetical external utility database is shown in below table.

| Item |  Profit|
| --- | --- |
| a | 4 |
| b | 3 |
|c | 6 |
|d | 2 |
|e | 5 |
|f | 2 |
|g | 3 |


__Note:__  Duplicate items must not exist in a transaction.

## 3. What is acceptable format of a utility databases in PAMI?

Each row in a utility database must contain the following information:
- set of items 
- total sum of utilties of all items in a transaction and 
- individual utility values of the items within a transaction.

All of the above three fields have to be seperated using the colan symbol.

A sample utility database, say  [sampleUtility.txt](sampleUtility.txt), is shown below:

a b c g:7:2 3 1 1 <br>
b c d e:10:3 2 3 2 <br>
a b c d:10:2 1 3 4 <br>
a c d f:7:3 2 1 2 <br>
a b c d g:9:3 1 2 1 2 <br>
c d e f:8:2 2 3 1 <br>
a b c d:6:2 1 1 2 <br>
a e f:5:1 2 2 <br>
a b c d:10:2 2 4 2 <br>
b c d e:9:3 2 2 2 <br>

## 4. What is the need for understanding the statistics of database?

The performance of a pattern mining algorithm primarily depends on the satistical nature of a database. Thus it is important to know the following details of a database:

* Total number of transactions (Database size)
* Total number of unique items in database
* Minimum lenth of transaction that existed in database
* Average length of all transactions that exists in database
* Maximum length of transaction that existed in database
* Minimum utility value exists in database
* Average utility exists in database
* Maximum utility exists in database
* Standard deviation of transaction length
* Variance in transaction length
* Sparsity of database

The below sample code prints the statistical details of a database.

```python 
import PAMI.extras.dbStats.UtilityDatabase as stats

obj = stats.UtilityDatabase('sampleUtility.txt', ' ')
obj.run()
obj.printStats()
```

## 5. What are the input parameters to a high utility pattern mining algorithm?

The input parameters to a frequent pattern mining algorithm are: 
* __Utility database__  <br> Acceptable formats:
> * String : E.g., 'utilityDatabase.txt'
> * URL  : E.g., https://u-aizu.ac.jp/~udayrage/datasets/utilityDatabases/utility_T10I4D100K.csv
> * DataFrame with the header titled 'Transactions', 'Utility' and 'TransactionUtility'

* __minUtil__  <br> specified in 
> * __count__
* __seperator__ <br> default seperator is '\t' (tab space)

## 6. How to store the output of a high utility pattern mining algorithm?
The patterns discovered by a high utility pattern mining algorithm can be saved into a file or a data frame.

## How to run the high utility pattern mining algorithms in a terminal?

* Download the PAMI source code from [Github](https://github.com/udayRage/PAMI/archive/refs/heads/main.zip).
* Unzip the PAMI source code folder.
```console
foo@bar: cd PAMI/highUtilityPattern/basic
```
* Execute the python program on their terminal using the following syntax:
```console
foo@bar: python3 algorithmName.py iFile outputFile minUtil seperator
```

__Example:__ python3 `EFIM.py` `inputFile.txt` `outputFile.txt` $20$ &nbsp; `' '`

## 7. How to execute a High utility pattern mining algorithm in a Jupyter Notebook?

- Install the PAMI package from the PYPI repository by executing the following command:   **pip3 install PAMI**
* Run the below sample code by making necessary changes

```python
import PAMI.highUtilityPattern.basic.EFIM as alg 

iFile = 'sampleUtility.txt'  #specify the input utility database <
minUtil = 20                #specify the minSupvalue
seperator = ' '               #specify the seperator. Default seperator is tab space. 
oFile = 'utilityPatterns.txt'   #specify the output file name

obj = alg.EFIM(iFile, minUtil, seperator) #initialize the algorithm 
obj.mine()                       #start the mining process 
obj.save(oFile)               #store the patterns in file 
df = obj.getPatternsAsDataFrame()     #Get the patterns discovered into a dataframe 
obj.printResults()                      #Print the stats of mining process
```

    High Utility patterns were generated successfully using EFIM algorithm
    Total number of High Utility Patterns: 11
    Total Memory in USS: 110927872
    Total Memory in RSS 149282816
    Total ExecutionTime in seconds: 0.0006177425384521484


The utilityPatterns.txt file contains the following patterns (*format:* pattern:utility):!cat utilityPatterns.txt


```python
!cat utilityPatterns.txt
```

    e	d	c:20 
    a	b	d:23 
    a	b	d	c:33 
    a	b	c:30 
    a	d:22 
    a	d	c:34 
    a	c:27 
    b	d:25 
    b	d	c:39 
    b	c:29 
    d	c:35 


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
      <th>Utility</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>e d c</td>
      <td>20</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a b d</td>
      <td>23</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a b d c</td>
      <td>33</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a b c</td>
      <td>30</td>
    </tr>
    <tr>
      <th>4</th>
      <td>a d</td>
      <td>22</td>
    </tr>
    <tr>
      <th>5</th>
      <td>a d c</td>
      <td>34</td>
    </tr>
    <tr>
      <th>6</th>
      <td>a c</td>
      <td>27</td>
    </tr>
    <tr>
      <th>7</th>
      <td>b d</td>
      <td>25</td>
    </tr>
    <tr>
      <th>8</th>
      <td>b d c</td>
      <td>39</td>
    </tr>
    <tr>
      <th>9</th>
      <td>b c</td>
      <td>29</td>
    </tr>
    <tr>
      <th>10</th>
      <td>d c</td>
      <td>35</td>
    </tr>
  </tbody>
</table>
</div>


