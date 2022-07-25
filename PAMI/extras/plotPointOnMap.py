import folium
import pandas as pd
import re

class plotPointOnMap:

    def __init__(self, inputPatterns, k=10, sep='\t'):
        self.inputPatterns = inputPatterns
        self.k = k
        self.sep = sep

    def findTopKPatterns(self):
        Database = []
        if isinstance(self.inputPatterns, pd.DataFrame):
            patterns = []
            i = self.inputPatterns.columns.values.tolist()
            if 'Transactions' in i:
                patterns = self.inputPatterns['Patterns'].tolist()
            for pattern in patterns:
                if isinstance(pattern, str):
                    pattern = [item for item in pattern.strip().split(self.sep)]
                Database.append(pattern)
        elif isinstance(self.inputPatterns, dict):
            for pattern in self.inputPatterns:
                if isinstance(pattern, str):
                    pattern = [item for item in pattern.strip().split(self.sep)]
                Database.append(pattern)

        elif isinstance(self.inputPatterns, str):
            with open(self.inputPatterns, 'r') as f:
                for line in f:
                    pattern = [s for s in line.strip().split(':')][0]
                    pattern = [item for item in pattern.strip().split(self.sep)]
                    Database.append(pattern)

        patterns = sorted(Database, key=lambda x: len(x), reverse=True)
        # return {patternId: patterns[patternId - 1] for patternId in range(1, int(self.k) + 1)}
        return patterns[:self.k]

    def convertPOINT(self, patterns):
        locations = pd.DataFrame(columns=['patternId', 'latitude', 'longitude'])
        patternId = 1
        for pattern in patterns:
            for item in pattern:
                location = item.split(' ')
                latitude = re.sub('[^0-9. ]', '', location[0])
                longitude = re.sub('[^0-9. ]', '', location[1])
                df = pd.DataFrame([patternId, latitude, longitude], index=locations.columns).T
                locations = locations.append(df, ignore_index=True)
            patternId += 1
        return locations



    def plotPointInMap(self):
        topKPatterns = self.findTopKPatterns()
        df = self.convertPOINT(topKPatterns)
        mmap = folium.Map(location=[35.39, 139.44], zoom_start=4)
        # df = pd.read_csv(inputFile)
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'beige', 'darkblue', 'darkgreen',
                  'cadetblue', 'darkpurple', 'white', 'pink','gray', 'black']
        for i, row in df.iterrows():
            mmap.add_child(folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                popup=row['patternId'],
                radius=3,
                color=colors[int(row['patternId']) - 1],
                fill=True,
                fill_color=colors[int(row['patternId']) - 1],
            ))
        return mmap

if __name__ == '__main__':
    obj = plotPointOnMap('visualizePatterns.csv')
    mmap = obj.plotPointInMap()
    mmap
