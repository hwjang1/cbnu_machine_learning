import pandas as pd
import random as rd
import numpy as np
from tqdm import tqdm

K = 2

print('-------- samples init --------')
Samples = []
Samples.append([0.0, 0.5])
Samples.append([1.0, 0.0])
Samples.append([0.0, 1.0])
Samples.append([1.0, 1.0])
Samples.append([2.0, 1.0])
Samples.append([1.0, 2.0])
Samples.append([2.0, 2.0])
Samples.append([3.0, 2.0])
Samples.append([6.0, 6.0])
Samples.append([6.0, 7.0])
Samples.append([7.0, 6.0])
Samples.append([7.0, 7.0])
Samples.append([7.0, 8.0])
Samples.append([8.0, 6.0])
Samples.append([8.0, 7.0])
Samples.append([8.0, 8.0])
Samples.append([8.0, 9.0])
Samples.append([9.0, 7.0])
Samples.append([9.0, 8.0])
Samples.append([9.0, 9.0])
print(Samples)


df = pd.DataFrame(Samples, columns=['x', 'y'])
print('-------- to dataframe --------')
print(df.head())



print('-------- centroid init --------')
centroids = []
for _ in range(K):
    d = df.iloc[_]
    centroids.append(np.array((d.x, d.y)))
print(centroids)



print('-------- calc centroid euclidean distance --------')
for i, centroid in enumerate(centroids):
    df['centroid%s' % i] = df.apply(lambda x: np.linalg.norm(np.array((x['x'], x['y'])) - centroid), axis=1)
print(df)


print('-------- "select_cluster" function init --------')
def select_cluster(rows):
    centroid_val = []
    for key in rows.keys():
        if 'centroid' in key:
            centroid_val.append({'key': key, 'val': rows[key]})
    return min(centroid_val, key=lambda item:item['val'])['key']


print('-------- select cluster --------')
df['cluster'] = df.apply(lambda x: select_cluster(x), axis=1)
print(df)



register_centroid = []
for loop_count in range(10):
    print('-------- loop centroid init --------')
    loop_centroids = []
    for i in range(K):
        loop_data = df[df['cluster'] == 'centroid%s' % i]
        loop_x = loop_data['x'].sum() / len(loop_data)
        loop_y = loop_data['y'].sum() / len(loop_data)
        loop_centroids.append(np.array((loop_x, loop_y)))
    if not np.array_equal(register_centroid, loop_centroids):
        register_centroid = loop_centroids
    else:
        break
    print('-------- loop calc centroid euclidean distance --------')
    for i, centroid in enumerate(loop_centroids):
        df['centroid%s' % i] = df.apply(lambda x: np.linalg.norm(np.array((x['x'], x['y'])) - centroid), axis=1)
    print('-------- loop select cluster --------')
    df['cluster'] = df.apply(lambda x: select_cluster(x), axis=1)


print('update centroid count : %s' % loop_count)

for i in range(K):
    print('centroid_%s value -> %s, %s' % (i, register_centroid[i][0], register_centroid[i][1]))
    print('centroid_%s members -> start')
    print(df[df['cluster'] == 'centroid%s' % i])
    print('centroid_%s members -> end')
    print('\n')
