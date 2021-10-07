import folium
import pandas as pd

def plotPointIntoMap(inputFile):
    mmap = folium.Map(location=[35.39, 139.44], zoom_start=5)
    df = pd.read_csv(inputFile)
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen',
              'cadetblue', 'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']
    for i, row in df.iterrows():
        folium.CircleMarker(
            location=(row['latitude'], row['longitude']),
            popup=row['patternId'],
            radius=3,
            # icon=folium.Icon(color=colors[int(row['patternId'])-1])
            color=colors[int(row['patternId']) - 1],
            fill=True,
            fill_color=colors[int(row['patternId']) - 1],
        ).add_to(mmap)
    return mmap