# Mining  Fuzzy Frequent Patterns in Fuzzy Transactional Databases

## 1. What is Fuzzy Frequent pattern mining?

Frequent pattern mining is a renowned data mining technique that aims to discover frequently occurring patterns in a (binary) transactional database. A fundamental limitation of this model is that it fails to discover interesting patterns that may exist in a quantitative (or non-binary) transactional database. When encountered with this limitation in the real-world applications, researchers try to find frequent patterns by converting a quantitative transactional database into a fuzzy transactional database using a set of fuzzy functions. The frequent patterns generated from the fuzzy transactional database are known as **fuzzy frequent patterns**.

Formally, fuzzy frequent pattern mining aims to discover all patterns that satisfy the user-specified **minimum support** (*minsup*) in a fuzzy transactional database. The **minSup** controls the minimum number of transactions that a pattern must appear in a database.

Reference: Lin, Chun-Wei & Li, Ting & Fournier Viger, Philippe & Hong, Tzung-Pei. (2015). A fast Algorithm for mining fuzzy frequent itemsets. Journal of Intelligent & Fuzzy Systems. 29. 2373-2379. 10.3233/IFS-151936. [link](https://dl.acm.org/doi/10.3233/IFS-151936)

## 2. What is a Fuzzy transactional database?

A fuzzy transactional database is a collection of transactions, where each transaction contains a set of fuzzy items and their respective fuzzy (or probability) values.  Please note that the fuzzy values of a fuzzy item will always lie between (0,1) or (0%, 100%).

Given a quantitative transactional database containing the items,  *a, b, c, d, e, f and g*, and  a set of fuzzy membership labels, Low (L), Medium (M), and High (H), a generated hypothetical fuzzy database is shown below.

| Transactions|                                     
| --- |                                              
| (a.L,0.2) (b.M,0.3) (c.H,0.1) (g.M,0.1) |                          
| (b.M,0.3) (c.H,0.2) (d.L,0.3) (e.H,0.2) |                          
| (a.L,0.2) (b.M,0.1) (c.H,0.3) (d.L,0.4) |                          
| (a.L,0.3) (c.H,0.2) (d.L,0.1) (f.M,0.2) |                          
| (a.L,0.3) (b.M,0.1) (c.H,0.2) (d.L,0.1) (g.M,0.2) |                    
| (c.H,0.2) (d.L,0.2) (e.H,0.3) (f.M,0.1) |                          
| (a.L,0.2) (b.M,0.1) (c.H,0.1) (d.L,0.2) |                          
| (a.L,0.1) (e.H,0.2) (f.M,0.2) |
| (a.L,0.2) (b.M,0.2) (c.H,0.4) (d.L,0.2) |
| (b.M,0.3) (c.H,0.2) (d.L,0.2) (e.H,0.2) |

__Note:__  Duplicate items must not exist in a transaction.

## 3. What is acceptable format of a fuzzy transactional database in PAMI?

Each row in a fuzzy transactional database must contain list of fuzzy items, colon as a seperator, and their list of fuzzy values. <br>
- Colon ':' must be used as a seperator to distinguish fuzzy items and their fuzzy values. This seperator is fixed and cannot be over-written by the users.
- fuzzy items and fuzzy values have to be seperated from each other with a delimitar. The default delimitar is 'tab space,' however, users can over-ride this default seperator.

A sample fuzzy transactional database file, say [fuzzyTransactionalDatabase.txt](fuzzyTransactionalDatabase.txt), is provided below:

a.L b.M c.H g.M:0.2 0.3 0.1 0.1 <br>
b.M c.H d.L e.H:0.13 0.2 0.3 0.2 <br>
a.L b.M c.H d.L:0.2 0.1 0.3 0.4 <br>
a.L c.H d.L f.M:0.3 0.2 0.1 0.2 <br>
a.L b.M c.H d.L g.M:0.3 0.1 0.2 0.1 0.2 <br>
c.H d.L e.H f.M:0.2 0.2 0.3 0.1 <br>
a.L b.M c.H d.L:0.2 0.1 0.1 0.2 <br>
a.L e.H f.M:0.1 0.2 0.2 <br>
a.L b.M c.H d.H:0.2 0.2 0.4 0.2 <br>
b.M c.H d.L e.H:0.3 0.2 0.2 0.2 <br>


For more information on how to create a fuzzy transactional database from a quantitative (or utility) transactional database, please refer to the manual [utility2FuzzyDB.pdf](utility2FuzzyDB.pdf)

## 4. What is the need for understanding the statistics of a fuzzy transactional database?

The performance of a pattern mining algorithm primarily depends on the satistical nature of a database. Thus, it is important to know the following details of a database:
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

obj = stats.FuzzyDatabase('fuzzyTransactionalDatabase.txt', ' ')
obj.run()
obj.printStats() 
```

## 5. What are the input parameters to be specified for a fuzzy frequent pattern mining algorithm?

The input parameters to a fuzzy frequent pattern mining algorithm are: 

* __Fuzzy transactional database__  <br> Acceptable formats:
> * String : E.g., 'fuzzyDatabase.txt'
> * URL  : E.g., https://u-aizu.ac.jp/~udayrage/datasets/fuzzyDatabases/fuzzy_T10I4D100K.csv
> * DataFrame with the header titled 'Transactions' and 'fuzzyValues'

* __minSup__  <br> specified in 
> * __count (beween 0 to length of a database)__ or 
> * [0, 1]
* __seperator__ <br> default seperator is '\t' (tab space)

## 6. How to store the output of a fuzzy frequent pattern mining algorithm?
The patterns discovered by a fuzzy frequent pattern mining algorithm can be saved into a file or a data frame.

## 7. How to run a fuzzy frequent pattern mining algorithms in a terminal?

* Download the PAMI source code from [Github](https://github.com/udayRage/PAMI).
* Unzip the PAMI source code folder.
* Enter into fuzzyFrequentPatterns folder.

```console
foo@bar: cd PAMI/fuzzyFrequentPatterns/basic
```

* Execute the python program on the terminal using the following syntax:

```console
foo@bar: python3 algorithmName.py inputFile outputFile minSup seperator
```


__Example:__ python3 `FFIMiner.py` `inputFile.txt` `outputFile.txt` `5` &nbsp; `' '`

## 8. How to execute a fuzzy frequent pattern mining algorithm in a Jupyter Notebook?

- Install the PAMI package from the PYPI repository by executing the following command:   **pip3 install PAMI**
* Run the below sample code by making necessary changes

```python
import PAMI.fuzzyFrequentPattern.basic.FFIMiner as alg

iFile = 'fuzzyTransactionalDatabase.txt'  # specify the input utility database 
minSup = 0.9  # specify the minSupvalue 
seperator = ' '  # specify the seperator. Default seperator is tab space. 
oFile = 'fuzzyPatterns.txt'  # specify the output file name

obj = alg.FFIMiner(iFile, minSup, seperator)  # initialize the algorithm 
obj.mine()  # start the mining process 
obj.save(oFile)  # store the patterns in file 
df = obj.getPatternsAsDataFrame()  # Get the patterns discovered into a dataframe 
obj.printResults()  # Print the stats of mining process
```

    Total number of Fuzzy Frequent Patterns: 7
    Total Memory in USS: 99786752
    Total Memory in RSS 137846784
    Total ExecutionTime in seconds: 0.0011456012725830078



```python
!cat fuzzyPatterns.txt
#format: fuzzyFrequentPattern:support
```

    b.M:1.23 
    b.M	c.H:0.9299999999999999 
    d.L:1.4999999999999998 
    d.L	c.H:1.2 
    a.L:1.5 
    a.L	c.H:1.0 
    c.H:1.9000000000000001 


The dataframe contains the following information:


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
      <td>b.M</td>
      <td>1.23</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b.M c.H</td>
      <td>0.9299999999999999</td>
    </tr>
    <tr>
      <th>2</th>
      <td>d.L</td>
      <td>1.4999999999999998</td>
    </tr>
    <tr>
      <th>3</th>
      <td>d.L c.H</td>
      <td>1.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>a.L</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>a.L c.H</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>c.H</td>
      <td>1.9000000000000001</td>
    </tr>
  </tbody>
</table>
</div>




## 9. Mining multiple fuzzy frequent itemsets using MFFIMiner

MFFIMiner retains all frequent regions of an item. For example, both `milk.Low` and `milk.High` may be frequent, but they cannot appear together in one pattern. The input must already contain transformed fuzzy memberships.

Each row uses `items:memberships`, with item labels and memberships in the same order. Memberships must be finite values in `[0, 1]`. The item and membership lists must have the same length, and fuzzy item labels must be unique within a transaction. Use `item.Region` labels to identify regions of the same base item. A third field containing a total utility or total membership is not accepted.

The following sample uses spaces as the separator. Save it as `fuzzyTransactions.txt`:

```text
milk.Low milk.High bread.High:0.8 0.2 0.9
milk.Low milk.High bread.High:0.3 0.7 0.6
milk.Low milk.High bread.High:0.6 0.4 0.8
milk.Low milk.High bread.High:0.1 0.9 0.4
```

Fuzzy support is the sum of the minimum membership of a pattern's items in each transaction. If an item is absent, that transaction contributes zero. In the sample, `milk.High bread.High` has support `0.2 + 0.6 + 0.4 + 0.4 = 1.6`.

An integer `minSup=1` sets an absolute fuzzy support threshold of 1. A float `minSup=0.25` sets the threshold to `0.25 * number_of_transactions`, which is also 1 for this four-row sample. Patterns with support equal to the threshold are included. Numeric strings follow the same rules: `'1'` is an absolute threshold, while `'1.0'` is a proportion of 1.0.

Use a PAMI installation that includes MFFIMiner. To install a source checkout, run `python3 -m pip install -e .` from the repository root. MFFIMiner runs on the CPU and does not require CUDA.

```python
from PAMI.fuzzyFrequentPattern.basic import MFFIMiner as alg

obj = alg.MFFIMiner('fuzzyTransactions.txt', minSup=1, sep=' ')
obj.mine()
patterns = obj.getPatterns()
patternsDF = obj.getPatternsAsDataFrame()
obj.save('mffiPatterns.txt')
obj.printResults()
```

The sample produces these five patterns:

| Pattern | Fuzzy support |
| --- | --- |
| bread.High | 2.7 |
| milk.High | 2.2 |
| milk.Low | 1.8 |
| bread.High milk.High | 1.6 |
| bread.High milk.Low | 1.8 |

`getPatterns()` returns a dictionary with tuples of sorted item labels as keys. `getPatternsAsDataFrame()` returns `Patterns` and `Support` columns. `save()` writes one pattern per line as `fuzzyItem<sep>fuzzyItem:support`, using the separator supplied to the miner.

The input can also be a URL or a DataFrame with `Transactions` and `fuzzyValues` columns. DataFrame cells may contain lists, or strings using the selected separator. The default separator is a tab (`'\t'`).

To run the sample from a terminal in the installed source checkout:

```console
python3 PAMI/fuzzyFrequentPattern/basic/MFFIMiner.py fuzzyTransactions.txt mffiPatterns.txt 1 ' '
```

`getRuntime()` reports elapsed mining time in seconds. `getMemoryRSS()` and `getMemoryUSS()` report process memory in bytes after mining; they do not report peak memory or memory used only by MFFIMiner. Low minimum support thresholds can produce many patterns and increase runtime and memory use.

See the [MFFIMiner notebook](https://github.com/UdayLab/PAMI/blob/main/notebooks/fuzzyFrequentPattern/basic/MFFIMiner.ipynb) for file, DataFrame, URL, and threshold comparison examples, and the [MFFIMiner API documentation](https://pami-1.readthedocs.io/en/latest/fuzzyFrequentPatternbasicMFFIMiner.html) for the class and methods.

Reference: Lin et al., [Efficient Mining of Multiple Fuzzy Frequent Itemsets](https://doi.org/10.1007/s40815-016-0246-1).
