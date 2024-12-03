# Mining Geo-referenced Fuzzy Periodic-Frequent Patterns in Geo-Referenced Fuzzy Temporal Databases

## 1. What is Geo-referenced Fuzzy Periodic-Frequent pattern mining?

Fuzzy Frequent Spatial Pattern mining aims to discover all Spatial fuzzy periodic patterns in a fuzzy database that have **support** no less than the user-specified **minimum support** (**minSup**) constraint,  **periodicity** no greater than user-specified **maximum periodicity** (**maxPer**) constraint and **distance** between two items is no less than **maximum distance** (**maxDist**). The **minSup** controls the minimum number of transactions that a pattern must appear in a database and the **maxPer** controls the maximum time interval within which a pattern must reappear in the database.

Reference: Veena et al., Mining Geo-referenced Fuzzy Periodic-Frequent Patterns in Geo-Referenced Fuzzy Temporal Databases, to be appeared in IEEE FUZZ 2022.

## 2. What is a fuzzy temporal database?

A fuzzy temporal database is a collection of transactions at a particular timestamp, where each transaction contains a timestamp, set of items, and its fuzzy values respectively. <br> 

A hypothetical fuzzy temporal database with items **_a, b, c, d, e, f and g_** and its fuzzy values are shown below:

|TS| Transactions|                                     
| --- | --- |                                          
| 1| (a.L,0.2) (b.M,0.3) (c.H,0.1) (g.M,0.1) |                          
| 2|(b.M,0.3) (c.H,0.2) (d.L,0.3) (e.H,0.2)|                          
| 3|(a.L,0.2) (b.M,0.1) (c.H,0.3) (d.L,0.4)|                          
| 4|(a.L,0.3) (c.H,0.2) (d.L,0.1) (f.M,0.2)|                          
| 5|(a.L,0.3) (b.M,0.1) (c.H,0.2) (d.L,0.1) (g.M,0.2)|                    
| 6|(c.H,0.2) (d.L,0.2) (e.H,0.3) (f.M,0.1)|                          
| 7|(a.L,0.2) (b.M,0.1) (c.H,0.1) (d.L,0.2) |                        
| 8|(a.L,0.1) (e.H,0.2) (f.M,0.2) |
| 9|(a.L,0.2) (b.M,0.2) (c.H,0.4) (d.L,0.2)|
| 10|(b.M,0.3) (c.H,0.2) (d.L,0.2) (e.H,0.2) |

__Note:__  Duplicate items must not exist in a transaction.

## 3. What is the acceptable format of a fuzzy temporal database in PAMI?

Each row in a fuzzy temporal database must contains list of fuzzy items, colon as a seperator, and their list of fuzzy values. <br>
- Colon ':' must be used as a seperator to distinguish fuzzy items and their fuzzy values. This seperator is fixed and cannot be over-written by the users.
- fuzzy items and fuzzy values have to be seperated from each other with a delimitar. The default delimitar is 'tab space,' however, users can over-ride this default seperator.

A sample fuzzy temporal database file, say [fuzzyTemporalDatabase.txt](fuzzyTemporalDatabase.txt), is provided below:

1 a.L b.M c.H g.M:0.2 0.3 0.1 0.1 <br>
2 b.M c.H d.L e.H:0.13 0.2 0.3 0.2 <br>
3 a.L b.M c.H d.L:0.2 0.1 0.3 0.4 <br>
4 a.L c.H d.L f.M:0.3 0.2 0.1 0.2 <br>
5 a.L b.M c.H d.L g.M:0.3 0.1 0.2 0.1 0.2 <br>
6 c.H d.L e.H f.M:0.2 0.2 0.3 0.1 <br>
7 a.L b.M c.H d.L g.M:0.3 0.1 0.2 0.1 0.2 <br>
8 b.M c.H d.L:0.2 0.1 0.1 0.2 <br>
9 a.L b.M c.H d.L g.M:0.3 0.1 0.2 0.1 0.2 <br>
10 b.M c.H d.L e.H:0.3 0.2 0.2 0.2<br>

For more information on how to create a fuzzy transactional database from a quantitative (or utility) transactional database, please refer to the manual [utility2FuzzyDB.pdf](utility2FuzzyDB.pdf)

## 4. What is a spatial database?

Spatial database contain the spatial (neighbourhood) information of items. It contains the items and its nearset neighbours satisfying the **maxDist** constraint. <br>
A hypothetical spatial database containing items **_a, b, c, d, e, f and g_** and neighbours respectively is shown below.

|  Item |  Neighbours |
| --- | --- |
| a | b, c, d |
| b | a, e, g |
| c | a, d |
| d | a, c |
| e | b, f |
| f | e, g |
| g | b, f |

## 5. What is the acceptable format of a spatial database in PAMI?

Spatial database contain the spatial (neighbourhood) information of items. It contains the items and its nearset neighbours satisfying the **maxDist** constraint. <br>
A hypothetical spatial database containing items **_a, b, c, d, e, f and g_** and neighbours respectively is shown below.

a b c d <br>
b a e g <br>
c a d <br>
d a c <br>
e b f <br>
f e g <br>
g b f <br>

For more information on how to create a neighborhood file for a given dataset, please refer to the manual of [creating neighborhood file](neighborhoodCreation.pdf).

## 6. What is the need for understanding the statistics of database of fuzzy temporal database?

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
import PAMI.extras.dbStats.FuzzyDatabase as stats

obj = stats.FuzzyDatabase('sampleInputFile.txt', ' ')
obj.run()
obj.printStats() 
```

## 7. What are the input parameters to be specified for a geo-referenced fuzzy periodic-frequent pattern mining algorithm?

The input parameters to a frequent pattern mining algorithm are: 
* __Fuzzy temporal database__  <br> Acceptable formats:
> * String : E.g., 'fuzzyDatabase.txt'
> * URL  : E.g., https://u-aizu.ac.jp/~udayrage/datasets/fuzzyDatabases/fuzzy_T10I4D100K.csv
> * DataFrame with the header titled 'Transactions', and 'fuzzyValues'

* __Spatial database__  <br> Acceptable formats:
> * String : E.g., 'spatialDatabase.txt'
> * URL  : E.g., https://u-aizu.ac.jp/~udayrage/datasets/fuzzyDatabases/neighbour_T10I4D100K.csv
> * DataFrame with the header titled 'item' and 'Neighbours'

* __minSup__ <br>
should be mentioned in 
> * __count (beween 0 to length of database)__ 
> * [0, 1]
* __maxPer__ <br>
should be mentioned in 
> * __count (beween 0 to length of database)__ 
> * [0, 1]
* __seperator__ <br> default seperator is '\t' (tab space)

## 8. How to store the output of a geo referenced fuzzy periodic-frequent pattern mining algorithm?
The patterns discovered by a geo-referenced fuzzy periodic frequent pattern mining algorithm can be saved into a file or a data frame.

## 9. How to run the geo referenced fuzzy periodic-frequent pattern algorithm in a terminal?

* Download the PAMI source code from github.
* Unzip the PAMI source code folder.
* Enter into fuzzySpatialPeriodicFrequentPattern folder.

```console
foo@bar: cd PAMI/fuzzySpatialPeriodicFrequentPattern/basic
```

* Execute the python program on the terminal using the following syntax:

```console
foo@bar: python3 algorithmName.py inputFile outputFile neighbourFile minSup maxPer seperator
```

__Example:__ python3 `FGPFPMiner.py` `inputFile.txt` `outputFile.txt` `neighbourFile.txt` `5` `3` &nbsp; `' '`

## 10. How to execute a geo-referenced fuzzy periodic frequent pattern mining algorithm in a Jupyter Notebook?

- Install the PAMI package from the PYPI repository by executing the following command:   **pip3 install PAMI**
* Run the below sample code by making necessary changes

```python
import PAMI.fuzzyGeoreferencedPeriodicFrequentPattern.basic.FGPFPMiner as alg

iFile = 'sampleFuzzyTemporal.txt'  # specify the input utility database <br>
minSup = 0.8  # specify the minSupvalue <br>
maxPer = 4
seperator = ' '
oFile = 'fuzzySpatialPeriodicFrequentPatterns.txt'  # specify the output file name<br>
nFile = 'sampleNeighbourFile.txt'  # specify the neighbour file of database <br>

obj = alg.FGPFPMiner(iFile, nFile, minSup, maxPer, seperator)  # initialize the algorithm <br>
obj.mine()  # start the mining process <br>
obj.save(oFile)  # store the patterns in file <br>
df = obj.getPatternsAsDataFrame()  # Get the patterns discovered into a dataframe <br>
obj.printResults()  # Print the stats of mining process
```

    Total number of Spatial Fuzzy Periodic-Frequent Patterns: 5
    Total Memory in USS: 98078720
    Total Memory in RSS 137539584
    Total ExecutionTime in ms: 0.0010845661163330078



```python
!cat fuzzySpatialPeriodicFrequentPatterns.txt

#format: fuzzyGeoreferencedPeriodicFrequentPattern:support
```

    e.H:0.8 
    b.M:1.0999999999999999 
    d.L:1.63 
    a.L:1.5 
    c.H:1.9999999999999998 


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
      <td>e.H</td>
      <td>0.8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b.M</td>
      <td>1.0999999999999999</td>
    </tr>
    <tr>
      <th>2</th>
      <td>d.L</td>
      <td>1.63</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a.L</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>c.H</td>
      <td>1.9999999999999998</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
