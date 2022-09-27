# Mining High-Utility Frequent Patterns in Utility Databases

## 1. What is High-Utility Frequent pattern mining?

High utility pattern mining model disregards the frequency information of a pattern in a database. However, in many real-world applications, an interestingness of a pattern is determined by both of its value and frequency. In this context, high utility frequent pattern mining was introduced to discover only those interesting patterns that had high value and occurred at least certain number of times in a database.

High utility frequent pattern mining aims to discover all the patterns with **_utility_** of pattern is no less than user-specified **_minimum utility_** (**_minutil_**) and **_support_** is no less than user-specified **_minimum support_** (**_minSup_**).

Reference: R. Uday Kiran, T. Yashwanth Reddy, Philippe Fournier-Viger, Masashi Toyoda, P. Krishna Reddy, Masaru Kitsuregawa: Efficiently Finding High Utility-Frequent Itemsets Using Cutoff and Suffix Utility. PAKDD (2) 2019: 191-203 [Link](https://link.springer.com/chapter/10.1007/978-3-030-16145-3_15)

## 2.  What is the utility database?

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

## 3. What is the acceptable format of a utility database in PAMI?

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

## 4. What is the need for understanding the statistics of a database?

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

The sample code is provided below:
```python
import PAMI.extras.dbStats.utilityDatabaseStats as stats 
obj = stats.utilityDatabaseStats('sampleUtility.txt', ' ')
obj.run() 
obj.printStats()
```

## 5. What are the input parameters to a high utility frequent pattern mining algorithm?

Algorithms to mine the high-utility patterns requires utility database, minUtil, and minSup (specified by user).
* Utility database can be provided in following formats:
> * String : E.g., 'utilityDatabase.txt'
> * URL  : E.g., https://u-aizu.ac.jp/~udayrage/datasets/utilityDatabases/utility_T10I4D100K.csv
> * In DataFrame format (dataframe variable with heading `Transactions`, `Utilities` and `TransactionUtility`
* __minUtil__  <br> specified in 
> * [0, 1]
* __minSup__  <br> specified in 
> * __count (beween 0 to length of a database)__ or 
> * [0, 1]
* __seperator__ <br> default seperator is '\t' (tab space)

## 6. How to store the output of a high-utility frequent pattern mining algorithm?

The patterns dicovered by a high utility frequent pattern mining algorithm can be saved into a file or a data frame.

## 7. How to execute a high-utility frequent pattern algorithm in a terminal?

* Download the PAMI source code from [Github](https://github.com/udayRage/PAMI/archive/refs/heads/main.zip).
* Unzip the PAMI source code folder.
* Enter into highUtilityFrequentPattern folder.
```console
foo@bar: cd PAMI/highUtilityFrequent/basic
```
* Execute the python program on their terminal using the following syntax:
```console
foo@bar: python3 algorithmName.py inputFile outputFile minUtil minSup seperator
```

__Example:__ python3 `HUFIM.py` `inputFile.txt` `outputFile.txt` $20$ &nbsp; $5$ &nbsp; `' '`

## 7. How to exeecute a high utility frequent pattern mining algorithm in a Jupyter Notebook?

- Install the PAMI package from the PYPI repository by executing the following command:   **pip3 install PAMI**
* Run the below sample code by making necessary changes


```python
import PAMI.highUtilityFrequentPatterns.basic.HUFIM as alg 

iFile = 'sampleUtility.txt'  #specify the input transactional database 
minUtil = 25                #specify the minUtil value 
minSup = 5                  #specify the minSup value 
seperator = ' '            #specify the seperator. Default seperator is tab space. 
oFile = 'utilityfrequentPatterns.txt'   #specify the output file name

obj = alg.HUFIM(iFile, minUtil, minSup, seperator) #initialize the algorithm 
obj.startMine()                       #start the mining process 
obj.save(oFile)               #store the patterns in file 
df = obj.getPatternsAsDataFrame()     #Get the patterns discovered into a dataframe 
obj.printResults()                      #Print the statistics of mining process
```

    High Utility Frequent patterns were generated successfully using HUFIM algorithm
    Total number of High Utility Frequent Patterns: 7
    Total Memory in USS: 81223680
    Total Memory in RSS 119382016
    Total ExecutionTime in seconds: 0.0004372596740722656



```python
!cat utilityfrequentPatterns.txt
# The format of the file is pattern:utility:support
```

    c	d:35:8 
    c	d	a:34:5 
    c	d	b:39:6 
    c	a:27:6 
    c	a	b:30:5 
    c	b:29:7 
    d	b:25:6 



```python
df
#The dataframe containing the patterns is shown below.
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
      <th>Support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>c d</td>
      <td>35</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>c d a</td>
      <td>34</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c d b</td>
      <td>39</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>c a</td>
      <td>27</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>c a b</td>
      <td>30</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>c b</td>
      <td>29</td>
      <td>7</td>
    </tr>
    <tr>
      <th>6</th>
      <td>d b</td>
      <td>25</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>


