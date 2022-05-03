# **[Home](index.html) | [Exercises](exercises.html) | [Real-world Examples](examples.html)**  

# Exercise 2: Mining frequent patterns in very large databases using multiple algorithms


### Task 1: Implementing multiple algorithms by varying minSup

    import pandas as pd
    result = pd.DataFrame(columns=['algorithm', 'minSup', 'patterns', 'runtime', 'memory'])

    from PAMI.frequentPattern.basic import FPGrowth as alg
    dataset = 'https://www.u-aizu.ac.jp/~udayrage/datasets/temporalDatabases/temporal_T10I4D100K.csv'
    minSupList = [0.01,0.02,0.03,0.04,0.05]
    algorithm = 'FPGrowth'
    sep = '\t'
    
    for minSup in minSupList:
        obj = alg.FPGrowth(dataset, minSup=minSup, sep=sep)
        obj.startMine()
        df = pd.DataFrame([algorithm, minSup, len(obj.getPatterns()), obj.getRuntime(), obj.getMemoryRSS()], index=result.columns).T
        result = result.append(df, ignore_index=True)
    
    from PAMI.frequentPattern.basic import ECLAT as alg
    dataset = 'https://www.u-aizu.ac.jp/~udayrage/datasets/temporalDatabases/temporal_T10I4D100K.csv'
    minSupList = [0.01,0.02,0.03,0.04,0.05]
    algorithm = 'ECLAT'
    
    for minSup in minSupList:
        obj = alg.ECLAT(dataset, minSup=minSup)
        obj.startMine()
        df = pd.DataFrame([algorithm, minSup, len(obj.getPatterns()), obj.getRuntime(), obj.getMemoryRSS()], index=result.columns).T
        result = result.append(df, ignore_index=True)

## Task 2: Visualizing the results of multiple algorithm

    import plotly.express as px
    fig = px.line(result, x='minSup', y='patterns', color='algorithm', markers=True)
    fig.show()
    fig = px.line(result, x='minSup', y='runtime', color='algorithm', markers=True)
    fig.show()
    fig = px.line(result, x='minSup', y='memory', color='algorithm', markers=True)
    fig.show()
