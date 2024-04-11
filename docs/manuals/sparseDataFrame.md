[Previous](denseDataFrame.html)|[Home](index.html)|[Next(DenseFormatDF.html)

## Sparse dataframe

### Description
A sparse dataframe is basically a (non-sparse) matrix in which the first column represents the row-identifier/timestamp, 
the second column represents the item, and the third column represents the value of the corresponding item.

### Format of a sparse data frame

      rowIdentifier/timestamp   Item1   Value

### An example
A sparse dataframe generated from the customer purchase database is as follows:

  timestamp | Item | Value
  ---------|-----|---
    1| Bread | 3
    1|Jam|1
    1|Butter|2
    2|Bread|7
    2|Jam|2
   ...|...|...
