[__<--__ Previous ](aboutPAMI.html)|[Home](installation.html)|[_Next_-->](organization.html)

## Neighborhood database

### Description
1. A neighborhood database is a collection of _geo-referenced items_ and their neighbors.
2. A geo-referenced item _j_ is said to be a neighbor of another geo-referenced item _i_ if the distance between them is no more than the user-specified _maximum distance_ threshold value.
   That is, if _distance(i,j) <=maximumDistance_, we say _j_ is a neighbor of _i_.
3. A sample neighborhood database generated from the set of geo-referenced items, I={a,b,c,d,e,f}, is shown in below table:

| Item |  Neighbors|
|------|-----------|
|Point(0 0)|Point(1 0)  Point(0 1)|
|Point(1 0)|Point(0 0)  Point(0 1) Point(2 0)|
|...|...|

### Rules to create a neighborhood database
1. Every row in the neighborhood file must contain only geo-referenced items.
2. First item in a row is the main geo-referenced item. Remaining items in a row represent the neighbors of main item.
3. All items in a row are seperated with a seperator, say tab space.
4. __Note:__ Every item must repeat only once in a row.

### Format to create a neighborhood database

    item<seperator>NeighboringItem1<seperator>NeighboringItem2<seperator>...

 
### An example

    item1   item3   item4   item10  
    item2   item3   item5   item11  ...
    ...

### Procedure to generate neighborhood file

#### Step 1: Import the program

```Python
from PAMI.extras.neighbours import  createNeighborhoodFileUsingEuclideanDistance as alg
```

#### Step2: Specify the parameters

```Python
inputLocationFile='geoReferencedInputFile.csv'  #name of the input file 
outputNeighborhoodFile='neighborhoodFile.csv'       #name of the output file
maximumDistance=10      #specify your own value
seperator='\t'      #default seperator.
```

#### Step 3: Call the program

```Python
alg.createNeighborhoodFileUsingEuclideanDistance(inputLocationFile,outputNeighborhoodFile,maximumDistance,seperator)
```
