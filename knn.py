import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression


df = pd.read_csv('moviesDatabase.csv')

features_taken=['results/features/0/genre_ids', 
            'results/features/0/production_companies/0', 
            'results/features/0/vote_average', 
            'results/features/0/vote_count',
           'results/features/0/budget']

x_dummy = pd.get_dummies(df.loc[:,features_taken])

x = df[features_taken].values
y = df['labels']

mean_error = []
std_error = []

n_range = [1,2,3,4,5,6,7,8,9,10]
for n in n_range:
    model = KNeighborsClassifier(n_neighbors=n,weights='uniform')
    temp = []
    kf = KFold(n_splits = 5)
    for train, test in kf.split(x):
        model.fit(x[train],y[train])
        ypred = model.predict(x[test])
        scores = cross_val_score(model, x,y, cv =5, scoring ='f1')
    mean_error.append(np.array(scores).mean())
    std_error.append(np.array(scores).std())
plt.errorbar(n_range, mean_error,yerr=std_error,linewidth = 3)
plt.xlabel('n')
plt.ylabel('F1 score')
#plt.show()


model = KNeighborsClassifier(n_neighbors=5, weights='uniform')
model.fit(x,y)
ypred = model.predict(x)
print(ypred)


print(confusion_matrix(y, ypred))
