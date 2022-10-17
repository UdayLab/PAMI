PAMI.fuzzyFrequentPatterns.basic package
========================================

Submodules
----------

PAMI.fuzzyFrequentPatterns.basic.FFIMiner module
------------------------------------------------

.. automodule:: PAMI.fuzzyFrequentPatterns.basic.FFIMiner
   :members:
   :undoc-members:
   :show-inheritance:

**Methods to execute code on terminal**

        Format:
                  >>>  python3 FFIMinerMiner.py <inputFile> <outputFile> <minSup> <separator>

        Example:
                  >>>  python3  FFIMinerMiner.py sampleTDB.txt output.txt 6

        .. note:: minSup will be considered in percentage of database transactions


**Importing this algorithm into a python program**

.. code-block:: python

        from PAMI.fuzzyFrequentPatterns import FFIMiner as alg

        obj = alg.FFIMiner("input.txt", 2)

        obj.startMine()

        fuzzyFrequentPatterns = obj.getPatterns()

        print("Total number of Fuzzy Frequent Patterns:", len(fuzzyFrequentPatterns))

        obj.savePatterns("outputFile")

        memUSS = obj.getMemoryUSS()

        print("Total Memory in USS:", memUSS)

        memRSS = obj.getMemoryRSS()

        print("Total Memory in RSS", memRSS)

        run = obj.getRuntime()

        print("Total ExecutionTime in seconds:", run)

**Credits:**

         The complete program was written by  Sai Chitra.B  under the supervision of Professor Rage Uday Kiran.


+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

