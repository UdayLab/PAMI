PAMI.coveragePatterns.basic package
===================================

Submodules
----------

PAMI.coveragePatterns.basic.CMine module
----------------------------------------

.. automodule:: PAMI.coveragePatterns.basic.CMine
   :members:
   :undoc-members:
   :show-inheritance:


**Methods to execute code on terminal**

        Format:
                  >>>  python3 CMine.py <inputFile> <outputFile> <minRF> <minCS> <maxOR> <'\t'>

        Example:
                  >>>  python3 CMine.py sampleTDB.txt patterns.txt 0.4 0.7 0.5 ','



**Importing this algorithm into a python program**

.. code-block:: python

            from PAMI.coveragePattern.basic import CMine as alg

            obj = alg.CMine(iFile, minRF, minCS, maxOR, seperator)

            obj.startMine()

            coveragePatterns = obj.getPatterns()

            print("Total number of coverage Patterns:", len(coveragePatterns))

            obj.save(oFile)

            Df = obj.getPatternsAsDataFrame()

            memUSS = obj.getMemoryUSS()

            print("Total Memory in USS:", memUSS)

            memRSS = obj.getMemoryRSS()

            print("Total Memory in RSS", memRSS)

            run = obj.getRuntime()

            print("Total ExecutionTime in seconds:", run)


**Credits:**

         The complete program was written by P.Likhitha  under the supervision of Professor Rage Uday Kiran.

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



PAMI.coveragePatterns.basic.CPPG module
---------------------------------------

.. automodule:: PAMI.coveragePatterns.basic.CPPG
   :members:
   :undoc-members:
   :show-inheritance:



**Methods to execute code on terminal**

        Format:
                  >>>  python3 CPPG.py <inputFile> <outputFile> <minRF> <minCS> <maxOR> <'\t'>

        Example:
                  >>>   python3 CPPG.py sampleTDB.txt patterns.txt 0.4 0.7 0.5 ','



**Importing this algorithm into a python program**

.. code-block:: python

            from PAMI.coveragePattern.basic import CPPG as alg

            obj = alg.CPPG(iFile, minRF, minCS, maxOR)

            obj.startMine()

            coveragePatterns = obj.getPatterns()

            print("Total number of coverage Patterns:", len(coveragePatterns))

            obj.save(oFile)

            Df = obj.getPatternsAsDataFrame()

            memUSS = obj.getMemoryUSS()

            print("Total Memory in USS:", memUSS)

            memRSS = obj.getMemoryRSS()

            print("Total Memory in RSS", memRSS)

            run = obj.getRuntime()

            print("Total ExecutionTime in seconds:", run)


**Credits:**

         The complete program was written by P.Likhitha  under the supervision of Professor Rage Uday Kiran.

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

