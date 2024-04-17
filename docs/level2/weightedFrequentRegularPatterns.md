# Mining Weighted Frequent Regular Patterns in Transactional Databases

## 1. What is weighted frequent regular pattern mining?

Weighted frequent regular pattern mining aims to discover all interesting patterns in a transactional database that have **weightedsupport** no less than the user-specified **weighted minimum support** (**minWS**) constraint and **regularity** no greater than the user-specified **maximum regularity** (**_regularity_**).  The **minWS** controls the minimum number of transactions that a pattern must appear in a database. The **regularity** controls the minimum weight of item. <br>

Reference: K. Klangwisan and K. Amphawan, "Mining weighted-frequent-regular itemsets from transactional database," 2017 9th International Conference on Knowledge and Smart Technology (KST), Chonburi, Thailand, 2017, pp. 66-71, doi: 10.1109/KST.2017.7886090.

## 2. What is the temporal database?

A temporal database is a collection of transactions at a particular timestamp, where each transaction contains a timestamp and a set of items. <br> A hypothetical temporal database containing the items **_a, b, c, d, e, f, and g_** as shown below

|tid| Transactions|
| --- | --- |
| 1 | a b c d |
| 2 | c e f |
| 3 | a b e f g |
| 4 | a b c f g |
| 5 | d e g |
| 6 | a b c e g | 
| 7 | a b c e |
| 8 | a b d e |
| 9 | b c e |
| 10 | a e g |

__Note:__  Duplicate items must not exist in a transaction.

## 3. What is acceptable format of a transactional databases in PAMI

Each row in a transactional database must contain only items. The frequent pattern mining algorithms in PAMI implicitly assume the row number of a transaction as its transactional-identifier to reduce storage and processing costs. A sample transactional database, say [sample.txt](sample.txt), is provided below.

1 a b c d <br>
2 c e f <br>
3 a b e f g <br>
4 a b c f g <br>
5 d e g <br>
6 a b c e g <br>
7 a b c e <br>
8 a b d e <br>
9 b c e <br>
10 a e g <br>

## 4. What is the Weighted database?

A weight database is a collection of items with their weights. <br> 
A hypothetical weight database, say  [WFRIWeightSample.txt](WFRIWeightSample.txt), containing the items **_a, b, c, d, e, f, and g_** as shown below

a 0.60  <br>
b 0.50  <br>
c 0.35  <br>
d 0.45  <br>
e 0.45  <br>
f 0.3  <br>
g 0.4 <br>

## 5. Understanding the statisctics of database

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
import PAMI.extras.dbStats.TemporalDatabase as stats

obj = stats.TemporalDatabase('WFRISample.txt', ' ')
obj.run()
obj.printStats() 
```

    Database size : 10
    Number of items : 7
    Minimum Transaction Size : 3
    Average Transaction Size : 4.0
    Maximum Transaction Size : 5
    Minimum period : 1
    Average period : 1.0
    Maximum period : 1
    Standard Deviation Transaction Size : 0.8944271909999159
    Variance : 0.8888888888888888
    Sparsity : 0.44285714285714284


The input parameters to a weighted frequent regular pattern mining algorithm are: 
* __Temporal database__  <br> Acceptable formats:
> * String : E.g., 'temporalDatabase.txt'
> * URL  : E.g., https://u-aizu.ac.jp/~udayrage/datasets/temporalDatabases/temporal_T10I4D100K.csv
> * DataFrame with the header titled with 'TS' and 'Transactions'

* __minWS__  <br> specified in 
> * __count (beween 0 to length of a database)__ or 
> * [0, 1]

* __regularity__  <br> specified in 
> * __count (beween 0 to length of a database)__ or 
> * [0, 1]

* __seperator__ <br> default seperator is '\t' (tab space)

## 5. How to store the output of a weighted frequent regular pattern mining algorithm?
The patterns discovered by a weighted frequent regular pattern mining algorithm can be saved into a file or a data frame.

## 6. How to run the weighted frequent regular pattern mining algorithms in a terminal?

* Download the PAMI source code from github.
* Unzip the PAMI source code folder and enter into weighted frequent regular pattern folder.
* Enter into weightedFrequentRegularPattern folder
* Enter into a specific folder and execute the  following command on terminal.

__syntax:__ python3 algorithmName.py `<path to the input file>` `<path to the output file>` `<path to the weight file>` `<minWS>` `<regularity>` `<seperator>`

## 7. Sample command to execute the WFRIM code in weightedFrequentRegularPattern folder

__Example:__ python3 `WFRIM.py` `inputFile.txt` `outputFile.txt` `weightSample.txt` `3` `2` `' '`

## 8. How to execute a weighted frequent regular pattern mining algorithm in a Jupyter Notebook?

- Install the PAMI package from the PYPI repository by executing the following command:   **pip3 install PAMI**
* Run the below sample code by making necessary changes


```python
from PAMI.weightedFrequentRegularPattern.basic import WFRIMiner as alg

iFile = 'WFRISample.txt'  #specify the input transactional database <br>
wFile = 'WFRIWeightSample.txt'  #specify the input transactional database <br>
minWS = 2  #specify the minSupvalue <br>
regularity = 3    #specify the minWeight value <br>
seperator = ' ' #specify the seperator. Default seperator is tab space. <br>
oFile = 'weightedFrequentRegularPatterns.txt'   #specify the output file name<br>

obj = alg.WFRIMiner(iFile, wFile, minWS, regularity, seperator) #initialize the algorithm <br>
obj.startMine()                       #start the mining process <br>
obj.save(oFile)               #store the patterns in file <br>
df = obj.getPatternsAsDataFrame()     #Get the patterns discovered into a dataframe <br>
obj.printResults()                      #Print the stats of mining process
```

    Weighted Frequent Regular patterns were generated successfully using WFRIM algorithm
    Total number of  Weighted Frequent Regular Patterns: 9
    Total Memory in USS: 99409920
    Total Memory in RSS 140251136
    Total ExecutionTime in ms: 0.00046825408935546875


The weightedFrequentRegularPatterns.txt file contains the following patterns (*format:* pattern:support): !cat weightedPatterns.txt


```python
!cat weightedFrequentRegularPatterns.txt
```

    c:[6, 2, 2.0999999999999996] 
    c	b:[5, 3, 2.125] 
    a:[7, 2, 4.2] 
    a	e:[5, 3, 2.625] 
    a	e	b:[5, 3, 2.5833333333333335] 
    a	b:[7, 2, 3.8500000000000005] 
    e:[8, 2, 3.6] 
    e	b:[6, 3, 2.8499999999999996] 
    b:[8, 2, 4.0] 


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
      <td>c</td>
      <td>[6, 2, 2.0999999999999996]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>c b</td>
      <td>[5, 3, 2.125]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a</td>
      <td>[7, 2, 4.2]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a e</td>
      <td>[5, 3, 2.625]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>a e b</td>
      <td>[5, 3, 2.5833333333333335]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>a b</td>
      <td>[7, 2, 3.8500000000000005]</td>
    </tr>
    <tr>
      <th>6</th>
      <td>e</td>
      <td>[8, 2, 3.6]</td>
    </tr>
    <tr>
      <th>7</th>
      <td>e b</td>
      <td>[6, 3, 2.8499999999999996]</td>
    </tr>
    <tr>
      <th>8</th>
      <td>b</td>
      <td>[8, 2, 4.0]</td>
    </tr>
  </tbody>
</table>
</div>


