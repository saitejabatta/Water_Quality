#Importing the required Libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from genetic_selection import GeneticSelectionCV
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np



#Reading the data
data = pd.read_csv('/content/dataset.csv')

#Data preprocessing
data.dropna(inplace=True)
nc = data.columns

#Encoding the data using LabelEncoder
le = LabelEncoder()
for i in nc:
  if data[i].dtype == object:
    data[i] = le.fit_transform(data[i])
  elif data[i].dtype == float:
    data[i] = data[i].astype(int)

#Splitting the dataset into traing and testing set
x = data.drop(['Target'],axis=1)
y = data['Target']
X_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state =142)

#Feature Extraction using Random Forest Classifier
s=SelectFromModel(RandomForestClassifier(n_estimators=20))
s.fit(X_train,y_train)
print(s.get_support())
X_train.columns[(s.get_support())]



#slecting features based on Random Forest Classifier
x1 = data[['pH', 'Nitrate', 'Chloride', 'Turbidity', 'Fluoride', 'Copper',
       'Odor', 'Manganese', 'Color']]
y1 = data['Target']

x1_train,x1_test,y1_train,y1_test=train_test_split(x1,y1,test_size=0.2,random_state=142)


#Genetic Algorithm for feature Extraction
estimator = DecisionTreeClassifier()
model = GeneticSelectionCV(estimator, cv=5,verbose=0,scoring='accuracy', max_features=5,n_population = 100,crossover_proba=0.5,mutation_proba=0.2,n_generations = 50, crossover_independent_proba=0.5,
                           mutation_independent_proba=0.04,tournament_size=3, n_gen_no_change=10,caching=True, n_jobs=-1)
model.fit(x1,y1)
print(x1.columns[model.support_])

#selecting features based on genetic algorithm
x2 = data[['pH', 'Nitrate', 'Chloride', 'Turbidity', 'Manganese']]
y2 = data['Target']

x2_train,y2_train,x2_test,y2_test=train_test_split(x2,y2,test_size=0.25,random_state=142)

#creating the Logistic Regression model and fitting the data
lr = LogisticRegression()
lr.fit(x1_train,y1_train)

lr.score(x1_train,y1_train)
lr.score(x1_test,y1_test)

lr.fit(x2,y)
lr.score(x2,y)


#Support Vector Machine model
clf1 = SVC(kernel='linear')
clf1.fit(X_train,y_train)
clf1.score(X_train,y_train)
clf1.score(x_test,y_test)


clf1.fit(x1_train,y1_train)
clf1.score(x1_train,y1_train)
clf1.score(x1_test,y1_test)

clf = SVC(kernel='linear')
clf.fit(x2,y2)

clf.score(x2,y)



#Multi Layer Perceptron
clas = MLPClassifier(hidden_layer_sizes=(150,100,50),max_iter=300,activation='relu',solver='adam')
clas.fit(X_train,y_train)

clas.score(X_train,y_train)
clas.score(x_test,y_test)

clas.fit(x1_train,y1_train)

clas.score(x1_train,y1_train)
clas.score(x1_test,y1_test)

clas.fit(x2,y2)
clas.score(x2,y2)
