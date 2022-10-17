PAMI.faultTolerantFrequentPattern.basic package
===============================================

Submodules
----------

PAMI.faultTolerantFrequentPattern.basic.FTApriori module
--------------------------------------------------------

.. automodule:: PAMI.faultTolerantFrequentPattern.basic.FTApriori
   :members:
   :undoc-members:
   :show-inheritance:



**Methods to execute code on terminal**

        Format:
                  >>>    python3 FTApriori.py <inputFile> <outputFile> <minSup> <itemSup> <minLength> <faultTolerance>
        Example:
                  >>>    python3 FTApriori.py sampleDB.txt patterns.txt 10.0 3.0 3 1

        .. note:: minSup will be considered in times of minSup and count of database transactions

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

         The complete program was written by  P.Likhitha under the supervision of Professor Rage Uday Kiran.

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


PAMI.faultTolerantFrequentPattern.basic.VBFTMine module
-------------------------------------------------------

.. automodule:: PAMI.faultTolerantFrequentPattern.basic.VBFTMine
   :members:
   :undoc-members:
   :show-inheritance:


**Methods to execute code on terminal**

        Format:
                  >>>  python3 VBFTMine.py <inputFile> <outputFile> <minSup> <itemSup> <minLength> <faultTolerance>
        Example:
                  >>>   python3 VBFTMine.py sampleDB.txt patterns.txt 10.0 3.0 3 1

        .. note:: minSup will be considered in times of minSup and count of database transactions

**Importing this algorithm into a python program**

.. code-block:: python

                import PAMI.faultTolerantFrequentPattern.basic.VBFTMine as alg

                obj = alg.VBFTMine(iFile, minSup, itemSup, minLength, faultTolerance)

                obj.startMine()

                faultTolerantFrequentPatterns = obj.getPatterns()

                print("Total number of Fault Tolerant Frequent Patterns:", len(faultTolerantFrequentPatterns))

                obj.save(oFile)

                Df = obj.getPatternInDataFrame()

                print("Total Memory in USS:", obj.getMemoryUSS())

                print("Total Memory in RSS", obj.getMemoryRSS())


**Credits:**

         The complete program was written by  P.Likhitha  under the supervision of Professor Rage Uday Kiran.

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

