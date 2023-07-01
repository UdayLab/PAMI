[__<--__ Return to home page](index.html)

## Steps to write an algorithm in PAMI

#### Step 1: Writing the text to appear in help()

Before writing any code using hash mark (#) write the text about explaining the algorithm and how to execute it.

```Python
# AlgorithmX aims to find interesting patterns in a particular database type.
#
# **Importing this algorithm into a python program**
# --------------------------------------------------------
#
# from PAMI.model.patternType import AlgorithmX as alg
# obj = alg.AlgorithmX(inputParameters)
# obj.startMine()
# interestingPatterns = obj.getPatterns()
# print("Total number of interesting patterns:", len(interestingPatterns))
# obj.savePatterns(oFile)
# memUSS = obj.getMemoryUSS()
# print("Total Memory in USS:", memUSS)
# memRSS = obj.getMemoryRSS()
# print("Total Memory in RSS", memRSS)
# run = obj.getRuntime()
# print("Total ExecutionTime in seconds:", run)
```

#### Step 2: Add the copyright information
```Python
__copyright__ = """
 Copyright (C)  2021 Rage Uday Kiran

     This program is free software: you can redistribute it and/or modify
     it under the terms of the GNU General Public License as published by
     the Free Software Foundation, either version 3 of the License, or
     (at your option) any later version.

     This program is distributed in the hope that it will be useful,
     but WITHOUT ANY WARRANTY; without even the implied warranty of
     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
     GNU General Public License for more details.

     You should have received a copy of the GNU General Public License
     along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
```

#### Step 3: Import the abstract class
```Python
from PAMI.PatternModel.PatternType import abstract as _ab
```

#### Step 4: Define the class file for an algorithm by inheriting the abstract file 

```Python
class AlgorithmX(_ab._patternModel):
    """
        :Description:  describe the algorithm
        :Reference: provide the reference of the algorithm with URL of the paper, if possible
        
        :param  inputParameters: parameterType :
                    description of the parameters. Parameters are the variables used in problem definition of the model.
                    
        :Attributes:
        
            attributeName: type
                describe the purpose of the attributes.
                
        **Methods to execute code on terminal**
        ----------------------------------------

            Format:
                      >>> python3 AlgorithmX.py <inputFile> <outputFile> <thresholdValues>

            Example:
                      >>>  python3 Apriori.py sampleDB.txt patterns.txt 10.0

            .. note:: minSup will be considered in percentage of database transactions

                            
        **Importing this algorithm into a python program**
        ----------------------------------------------------

        .. code-block:: python
    
            from PAMI.model.patternType import AlgorithmX as alg
            obj = alg.AlgorithmX(inputParameters)
            obj.startMine()
            interestingPatterns = obj.getPatterns()
            print("Total number of interesting patterns:", len(interestingPatterns))
            obj.savePatterns(oFile)
            memUSS = obj.getMemoryUSS()
            print("Total Memory in USS:", memUSS)
            memRSS = obj.getMemoryRSS()
            print("Total Memory in RSS", memRSS)
            run = obj.getRuntime()
            print("Total ExecutionTime in seconds:", run)
            
        **Credits:**
        -------------
             The complete program was written by PersonX  under the supervision of Professor Y.        
    """
```
#### Step 5: Define the necessary functions
```Python
   def functionName(self, inputParameters):
    """
    Explain the purpose of this function.

    :param value: user specified minSup value

    :return: converted type
    """
```

#### Step 6: Mandatory functions
```Python
    def startMine(self):
        """
            Pattern mining process will start from here
        """
        #Write your code below.
        #...
        
    def getMemoryUSS(self):
        """Total amount of USS memory consumed by the mining process will be retrieved from this function

        :return: returning USS memory consumed by the mining process

        :rtype: float
        """

        return self._memoryUSS

    def getMemoryRSS(self):
        """Total amount of RSS memory consumed by the mining process will be retrieved from this function

        :return: returning RSS memory consumed by the mining process

        :rtype: float
        """

        return self._memoryRSS

    def getRuntime(self):
        """Calculating the total amount of runtime taken by the mining process

        :return: returning total amount of runtime taken by the mining process

        :rtype: float
        """

        return self._endTime - self._startTime

    def getPatternsAsDataFrame(self):
        """Storing final frequent patterns in a dataframe

        :return: returning frequent patterns in a dataframe

        :rtype: pd.DataFrame
        """

        dataFrame = {}
        data = []
        for a, b in self._finalPatterns.items():
            data.append([a.replace('\t', ' '), b])
            dataFrame = _ab._pd.DataFrame(data, columns=['Patterns', 'Support'])
        # dataFrame = dataFrame.replace(r'\r+|\n+|\t+',' ', regex=True)
        return dataFrame

    def save(self, outFile):
        """Complete set of frequent patterns will be loaded in to a output file

        :param outFile: name of the output file

        :type outFile: file
        """
        self._oFile = outFile
        writer = open(self._oFile, 'w+')
        for x, y in self._finalPatterns.items():
            s1 = x.strip() + ":" + str(y)
            writer.write("%s \n" % s1)

    def getPatterns(self):
        """ Function to send the set of frequent patterns after completion of the mining process

        :return: returning frequent patterns

        :rtype: dict
        """
        return self._finalPatterns

    def printResults(self):
        print("Total number of Frequent Patterns:", len(self.getPatterns()))
        print("Total Memory in USS:", self.getMemoryUSS())
        print("Total Memory in RSS", self.getMemoryRSS())
        print("Total ExecutionTime in ms:", self.getRuntime())

    
```

#### Step 7: Main function to call an algorithm from a terminal
```Python
if __name__ == "__main__":
    _ap = str()
    #Write your text below
```