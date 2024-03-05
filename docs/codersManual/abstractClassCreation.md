[__<--__ Return to home page](index.html)

## Creation of abstract.py file

Abstract file is crucial to maintain consistant terms across multiple algorithms designed for a finding interesting patterns of a particular time.
We now discuss the step-by-step approach to create the abstract.py file

__Step 1:__ Import the abstract base class and other necessary libraries
```Python
from abc import ABC as _ABC, abstractmethod as _abstractmethod
import time as _time
import csv as _csv
import pandas as _pd
from collections import defaultdict as _defaultdict
from itertools import combinations as _c
import os as _os
import os.path as _ospath
import psutil as _psutil
import sys as _sys
import validators as _validators
from urllib.request import urlopen as _urlopen
import functools as _functools

#Import other libraries if necessary
```

__Step 2:__ Define the class and provide the explanation of variables and methods used in this file

```Python
class _theoreticalNameOfThePatterns(_ABC):
    """
    Describe the purpose of this abstract file.
    
    Attributes:
    -----------
        attributeNate : type (str/int/boolean/double/float)
        Text explaining the purpose of the attribute
        
    Methods:
    --------
        methodName()
            purpose of the method.    
    """
```

__Step 3:__ Define __init__ method
```Python
    def __init__(self, inputVariables):
        """
            Describe the function with input parameters.
            
            :param inputVariables: Input file name or path of the input file
            :type inputVariables: str or DataFrame
        """
```

__Step 4:__ Define @abstractethod function for each method
```Python
    @_abstractmethod
    def method(self):
        """Purpose of this method"""
        pass
```

