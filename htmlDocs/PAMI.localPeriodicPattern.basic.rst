PAMI.localPeriodicPattern.basic package
=======================================

Submodules
----------

PAMI.localPeriodicPattern.basic.LPPGrowth module
------------------------------------------------

.. automodule:: PAMI.localPeriodicPattern.basic.LPPGrowth
   :members:
   :undoc-members:
   :show-inheritance:


**Methods to execute code on terminal**

        Format:
                  >>>   python3 LPPMGrowth.py <inputFile> <outputFile> <maxPer> <minSoPer> <minDur>
        Example:
                  >>>   python3 LPPMGrowth.py sampleDB.txt patterns.txt 3 4 5

        .. note:: minSup will be considered in percentage of database transactions


**Importing this algorithm into a python program**

.. code-block:: python

            from PAMI.localPeriodicPattern.basic import LPPGrowth as alg

            obj = alg.LPPGrowth(iFile, maxPer, maxSoPer, minDur)

            obj.startMine()

            localPeriodicPatterns = obj.getPatterns()

            print(f'Total number of local periodic patterns: {len(localPeriodicPatterns)}')

            obj.savePatterns(oFile)

            Df = obj.getPatternsAsDataFrame()

            memUSS = obj.getMemoryUSS()

            print(f'Total memory in USS: {memUSS}')

            memRSS = obj.getMemoryRSS()

            print(f'Total memory in RSS: {memRSS}')

            runtime = obj.getRuntime()

            print(f'Total execution time in seconds: {runtime})

**Credits:**

         The complete program was written by So Nakamura  under the supervision of Professor Rage Uday Kiran.

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



PAMI.localPeriodicPattern.basic.LPPMBreadth module
--------------------------------------------------

.. automodule:: PAMI.localPeriodicPattern.basic.LPPMBreadth
   :members:
   :undoc-members:
   :show-inheritance:


**Methods to execute code on terminal**

        Format:
                  >>>  python3 LPPMBreadth.py <inputFile> <outputFile> <maxPer> <minSoPer> <minDur>
        Example:
                  >>>  python3 LPPMBreadth.py sampleDB.txt patterns.txt 0.3 0.4 0.5

        .. note:: minSup will be considered in percentage of database transactions


**Importing this algorithm into a python program**

.. code-block:: python

        from PAMI.localPeriodicPattern.basic import LPPMBreadth as alg

        obj = alg.LPPMBreadth(iFile, maxPer, maxSoPer, minDur)

        obj.startMine()

        localPeriodicPatterns = obj.getPatterns()

        print(f'Total number of local periodic patterns: {len(localPeriodicPatterns)}')

        obj.savePatterns(oFile)

        Df = obj.getPatternsAsDataFrame()

        memUSS = obj.getMemoryUSS()

        print(f'Total memory in USS: {memUSS}')

        memRSS = obj.getMemoryRSS()

        print(f'Total memory in RSS: {memRSS}')

        runtime = obj.getRuntime()

        print(f'Total execution time in seconds: {runtime})

**Credits:**

         The complete program was written by  So Nakamura  under the supervision of Professor Rage Uday Kiran.

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



PAMI.localPeriodicPattern.basic.LPPMDepth module
------------------------------------------------

.. automodule:: PAMI.localPeriodicPattern.basic.LPPMDepth
   :members:
   :undoc-members:
   :show-inheritance:


**Methods to execute code on terminal**

        Format:
                  >>>  python3 LPPMDepth.py <inputFile> <outputFile> <maxPer> <minSoPer> <minDur> <sep>
        Example:
                  >>>  python3 LPPMDepth.py sampleDB.txt patterns.txt 3 4 5

        .. note:: minSup will be considered in percentage of database transactions


**Importing this algorithm into a python program**

.. code-block:: python

        from PAMI.localPeriodicPattern.basic import LPPMDepth as alg

        obj = alg.LPPMDepth(iFile, maxPer, maxSoPer, minDur)

        obj.startMine()

        localPeriodicPatterns = obj.getPatterns()

        print(f'Total number of local periodic patterns: {len(localPeriodicPatterns)}')

        obj.savePatterns(oFile)

        Df = obj.getPatternsAsDataFrame()

        memUSS = obj.getMemoryUSS()

        print(f'Total memory in USS: {memUSS}')

        memRSS = obj.getMemoryRSS()

        print(f'Total memory in RSS: {memRSS}')

        runtime = obj.getRuntime()

        print(f'Total execution time in seconds: {runtime})

**Credits:**

         The complete program was written by  So Nakamura  under the supervision of Professor Rage Uday Kiran.

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


