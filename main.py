from pymongo import MongoClient
import requests
from pprint import pprint
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.cluster import KMeans


def graph():
  data = pd.read_csv('db.csv')
  


#  sn.scatterplot('temperature', 'attendance',  hue='promixity_to_exam', data=data);
#  plt.title('Teemperature to Attendance Coloured by Proximity to examinations');
#  fig = plt.axes()
#  fig.set_facecolor('grey')
#  plt.savefig("Scatterplot.png")


  sn.countplot(x='teacher', data=data);
  plt.title('Distribution of Classes by Teachers');
  plt.savefig("CountPlot1.png");

  sn.countplot(x='subject', data=data);
  plt.title('Distribution of Subjects by Teachers');
  plt.savefig("CountPlot2.png");


  corr = data[['_id', 'date', 'time', 'temperature', 'humidity', 'subject', 'teacher', 'department', 'section', 'semester', 'promixity_to_exam', 'distm10', 'distl10', 'attendance']].corr()
  mask = np.array(corr)
  mask[np.tril_indices_from(mask)] = False
  fig,ax= plt.subplots()  
  fig.set_size_inches(20,10)
  plt.title('Correlation between various factors shown by a heatmap');
  sn.heatmap(corr, mask=mask,vmax=.9, square=True,annot=True, cmap="YlGnBu")
  plt.savefig("heatmap_plot.png")



  data.hist('attendance', bins=35);
  plt.title('Distribution of Attendance');
  plt.xlabel('Attendance');
  plt.savefig("Hist_attendance.png")

  data.hist('distm10');
  plt.title('Students living at a distance greater than 10 km');
  plt.xlabel('Number of Students');
  plt.savefig("Hist_number.png")

  data.hist('distl10');
  plt.title('Students living at a distance lesser than 10 km');
  plt.xlabel('Number of Students');
  plt.savefig("Hist_num.png")



  ax = sn.boxplot(x="temperature", y="attendance", data=data)
  plt.savefig("boxplot1.png")

  ax = sn.boxplot(x="humidity", y="attendance", data=data)
  plt.savefig("boxplot2.png")
  
  kmeans=KMeans(n_clusters=2)

  data=data.drop('section',axis=1)
  data=data.drop('department',axis=1)
  data=data.drop('teacher',axis=1)
  data=data.drop('subject',axis=1)
  data=data.drop('_id',axis=1)
  data=data.drop('date',axis=1)
  data=data.drop('time',axis=1)

  kmeans.fit(data)

  pred=kmeans.predict(data)

  kmeans.inertia_

  kmeans.score(data)

  SSE=[]

  for cluster in range(1,20):
    kmeans=KMeans(n_jobs=-1,n_clusters=cluster)
    kmeans.fit(data)
    SSE.append(kmeans.inertia_)

  frame= pd.DataFrame({'Cluster':range(1,20),'SSE':SSE})

  frame.head()

  plt.figure(figsize=(12,6))
  plt.plot(frame['Cluster'],frame['SSE'],marker='o')
  #plt.savefig("Kmeans1.png")

  from sklearn.preprocessing import StandardScaler
  scaler=StandardScaler()
  data_scaled=scaler.fit_transform(data)

  SSE_scaled=[]

  pd.DataFrame(data_scaled).describe()

  for cluster in range(1,20):
    kmeans=KMeans(n_jobs=-1,n_clusters=cluster)
    kmeans.fit(data_scaled)
    SSE_scaled.append(kmeans.inertia_)

  frame_scaled=pd.DataFrame({'Cluster':range(1,20),'SSE':SSE_scaled})
  plt.plot(frame_scaled['Cluster'],frame_scaled['SSE'],marker="o")
  plt.xlabel('Clusters')
  plt.ylabel('SSE')
  plt.savefig("Kmeans2.png")





client=MongoClient("mongodb+srv://admin_project:project@clusterlarkai1.idztf.mongodb.net/student?retryWrites=true&w=majority")
db=client.get_database('studentdb')
records=db.student

db=list(records.find())


csv_columns=['_id','date','time','temperature','humidity','male','female','subject','teacher','department','section','semester','promixity_to_exam','distm10','distl10','holiday','attendance']


csv_file='db.csv'

with open(csv_file,'w') as csvfile:
  writer=csv.DictWriter(csvfile,fieldnames=csv_columns)
  writer.writeheader()
  for data in db:
    writer.writerow(data)


graph()
