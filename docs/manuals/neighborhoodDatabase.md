[__<--__ Return to home page](index.html)

## Creation of neighborhood file from a geo-referenced (or spatial) database

This page describes the process to generate a neighborhood file for a given geo-referenced database.
The neighborhood file contains the information about an item and its neighboring items. An item _j_ is said to be a neighbor of item _i_ if the distance between them is no more than the user-specified threshold value. 
That is, if _distance(i,j) <=maximumDistance_, then we say _j_ is a neighborhood of _i_.

### 1. Format of Geo-referenced database
- A geo-referenced item is a spatial item, such as Point, Line, Polygon, and Pixel.
- Every line in this file contains a geo-referenced item.
- The format of this file is

      geoReferencedItem1
      geoReferencedItem2
      ...
      geoReferencedItemN 

### 2. An example of a geo-referenced database

      Point(0 0)
      Point(0 1)
      Point(1 1)
      ...
### 3. Format of a Neighborhood file
Each row in the neighborhood file contains an item and its neighbors seperated by a seperator, say tab space.
The format of the neighborhood file is as follows:

    item<seperator>NeighboringItem1<seperator>NeighboringItem2<seperator>...
The first column represents an item, while remaining columns represent the neighboring items of the item that exists in first column.

### 4. An example of a neighborhood file

    item1   item3   item4   item10  
    item2   item3   item5   item11  ...
    ...

The first line informs us that the item3, item4 and item10 are spatial neighbors of item1.
The second line also informs us that item3, item5 and item11 are spatial neighbors of item2.

### 5. Procedure to generate neighborhood file

#### Step 1: Import the program

```Python
from PAMI.extras.neighbours import  createNeighborhoodFileUsingEuclideanDistance as alg
```

#### Step2: Specify the parameters

```Python
inputFile='geoReferencedInputFile.csv'  #name of the input file 
outputFile='neighborhoodFile.csv'       #name of the output file
maximumDistance=10      #specify your own value
seperator='\t'      #default seperator. 

```

#### Step 3: Call the program

```Python
alg.createNeighborhoodFileUsingEuclideanDistance(inputFile,outputFile,maximumDistance,seperator)
```