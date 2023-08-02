[__<--__ Return to home page](index.html)

## Using Abstract Base Classes (ABC)

An abstract class acts as a blueprint for the algorithms developed for a theoretical pattern model.
It allows the coders to create a set of methods that must be created within any algorithm
built from the abstract class. 

In PAMI, abstract base class provides a common Application Program Interface(API) for a set of algorithms (or subclasses).

### Default file name of an Abstract class
In PAMI, the file name for creating an abstract base class is "abstract.py"

### Creating an abstract class

```Python
#Step 1: import ABC 
from abc import ABC as _ABC, abstractmethod as _abstractmethod

#Step 2: Create an abstract class with ABC as an input parameter
class _theoreticalPattern(ABC):
    #Step 3: declare an abstract method

    @abstractmethod
    def methodName(self):
        pass
    #Step 4: declaring an abstract variable
    @property
    @abstractmethod
    def variableName(self):
        pass
```
### Using an abstract class

```Python
#Step 1: Importing an abstract class
import abstract as _ab

#Step 2: Using an abstract class in an algorithm (or subclass)
class patternMiningAlgorithm(_ab._theoreticalPattern):

    #Step3: declare your variables and write your code here.

``'