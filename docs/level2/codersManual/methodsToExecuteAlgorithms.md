[__<--__ Return to home page](index.html)

## Methods to execute an algorithm in PAMI

The users can execute the algorithms in PAMI in the following two ways:
1. Direct execution of an algorithm from a terminal

```terminal
Python algorithmName <inputParameters>
```
  Every algorithm must include __main__ function. In other words, every algorithm must have the following statement.
```Python
if __name__ == "__main__":
    _ap = str()
```  


2. Calling the algorithm via a python program 

Every algorithm's class file must have a constructor that reads the input parameters.
This constructor file is mainly provided in the abstract class for reusing.
If any person wants to create their own constructor, then the format is given below:

```Python
class AlgorithmName(abstractClass):
    def __init__(self, inputParameters):
        #write your code here.    
```