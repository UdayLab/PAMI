PAMI.uncertainCorrelatedPattern.basic package
=============================================

Submodules
----------

PAMI.uncertainCorrelatedPattern.basic.CFFI module
-------------------------------------------------

.. automodule:: PAMI.uncertainCorrelatedPattern.basic.CFFI
   :members:
   :undoc-members:
   :show-inheritance:

**Methods to execute code on terminal**

        Format:
                  >>>   python3 CFFI.py <inputFile> <outputFile> <minSup> <ratio>
        Example:
                  >>>   python3 CFFI.py sampleTDB.txt output.txt 2 0.2


**Importing this algorithm into a python program**

.. code-block:: python

        from PAMI.uncertainCorrelatedPattern.basic import CFFI as alg

        obj = alg.CFFI("input.txt", 2, 0.4)

        obj.startMine()

        Patterns = obj.getPatterns()

        print("Total number of Correlated Fuzzy Frequent Patterns:",  len(Patterns))

        obj.savePatterns("outputFile")

        memUSS = obj.getMemoryUSS()

        print("Total Memory in USS:",  memUSS)

        memRSS = obj.getMemoryRSS()

        print("Total Memory in RSS",  memRSS)

        run = obj.getRuntime

        print("Total ExecutionTime in seconds:",  run)

**Credits:**

         The complete program was written by  Sai Chitra B  under the supervision of Professor Rage Uday Kiran.

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

