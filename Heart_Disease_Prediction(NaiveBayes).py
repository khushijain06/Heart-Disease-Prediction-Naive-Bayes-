import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
dataset = pd.read_csv('/content/dataset.csv')
print(dataset.head())
X = dataset.iloc[:,:13].values
y = dataset['target'].values
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 90)
# Feature Scaling to bring the variable in a single scale
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.naive_bayes import GaussianNB
nvclassifier = GaussianNB()
nvclassifier.fit(X_train, y_train)
y_pred = nvclassifier.predict(X_test)
#print(y_pred)
#lets see the actual and predicted value side by side
y_compare = np.vstack((y_test,y_pred)).T
#actual value on the left side and predicted value on the right hand side
#printing the top 5 values
y_compare[:5,:]
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test, y_pred)
print(cm)
from sklearn.metrics import ConfusionMatrixDisplay
cm_plot = ConfusionMatrixDisplay(cm , display_labels= [1,0])
cm_plot.plot()
plt.show()
import sklearn.metrics as metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test,y_pred))
print("Recall:",metrics.recall_score(y_test,y_pred))
print("Specificity:",metrics.recall_score(y_test,y_pred))
print("F1-score:", metrics.f1_score(y_test,y_pred))
#import seaborn as sns
#import matplotlib.pyplot as plt 
#plt.figure(figsize=(8,6))
#sns.heatmap(cm,annot=True , fmt ='d',cmap ='Blues',xticklabels=['Predicted 1','Predicted 0'], yticklabels=['Actual 1','Actual 0'])
#plt.xlabel('Predicted')
#plt.ylabel('Actual')
#plt.title('Confusion Matrix')
#plt.show()