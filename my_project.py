import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv('Country-data.csv',low_memory=False)

#import vs export
x=dataset.iloc[:,[2,4]].values

wcss=[]
from sklearn.cluster import KMeans
for i in range(1,15):
    kmean=KMeans(n_clusters=i)
    kmean.fit(x)
    wcss.append(kmean.inertia_)
    
    
#plt.plot(range(1,15),wcss)
    
kmean=KMeans(n_clusters=3)
kmean.fit(x)
y_pred=kmean.predict(x)

tunisian_economie=kmean.predict([[50.5,55.3]])
print(tunisian_economie)

#tunisia has a bad economie

plt.figure()
plt.scatter(x[y_pred == 0, 0], x[y_pred == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(x[y_pred == 1, 0], x[y_pred == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(x[y_pred == 2, 0], x[y_pred == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(kmean.cluster_centers_[:, 0], kmean.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of import/export')
plt.xlabel('export')
plt.ylabel('import')
plt.legend()
plt.show()


#income vs inflation

x=dataset.iloc[:,[5,6]].values

wcss=[]
from sklearn.cluster import KMeans
for i in range(1,15):
    kmean=KMeans(n_clusters=i)
    kmean.fit(x)
    wcss.append(kmean.inertia_)
    
kmean=KMeans(n_clusters=2)
kmean.fit(x)
y_pred=kmean.predict(x)


tunisian_economie_for_person=kmean.predict([[10400,3.82]])
print(tunisian_economie_for_person)

#tunisia has a bad tunisian_economie_for_person
plt.figure()
plt.scatter(x[y_pred == 0, 0], x[y_pred == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(x[y_pred == 1, 0], x[y_pred == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(kmean.cluster_centers_[:, 0], kmean.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of income vs inflation')
plt.xlabel('income')
plt.ylabel('inflation')
plt.legend()
plt.show()







#health vs life_expec

x=dataset.iloc[:,[3,7]].values

wcss=[]
from sklearn.cluster import KMeans
for i in range(1,15):
    kmean=KMeans(n_clusters=i)
    kmean.fit(x)
    wcss.append(kmean.inertia_)

#plt.plot(range(1,15),wcss)


kmean=KMeans(n_clusters=3)
kmean.fit(x)
y_pred=kmean.predict(x)

tunisian_health=kmean.predict([[6.21,76.9]])
print(tunisian_health)
#tinisia has meduim health care

plt.figure()
plt.scatter(x[y_pred == 0, 0], x[y_pred == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(x[y_pred == 1, 0], x[y_pred == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(x[y_pred == 2, 0], x[y_pred == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(kmean.cluster_centers_[:, 0], kmean.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of health vs life_expec')
plt.xlabel('health')
plt.ylabel('life_expec')
plt.legend()
plt.show()


