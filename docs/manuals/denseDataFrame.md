[Previous](timeSeries.html)|[Home](index.html)|[Next](sparseDataFrame.html)


## Dense dataframe

### Description
A dense dataframe is basically a  matrix in which the first column represents the row-identifier/timestamp
and the remaining columns represent the items and their values. T

### Format of a dense dataframe 

      rowIdentifier/timestamp   Item1   Item2   ... ItemN

### An example 

  timestamp | Bread | Jam | Butter | Books | Pencil
  ---------|-----|---|------|---|------
    1| 3 | 1| 2|0 |0
    2|7|2|0|10|20
    3|0|0|3|0|0
    4|4|0|0|0|0

In the above dataframe (or table), the first transaction (or row) provides the information that a customer has purchased the 3 packets 
of Bread, 1 bottle of Jam, 3 packets of Butter at the timestamp of 1. The second transaction provides the information
that a customer has purchased 7 packets of Bread, 2 bottles of Jam, 10 Books and 20 Pencils. Similar arguments can be 
made for the remaining transactions in the above dataframe.
