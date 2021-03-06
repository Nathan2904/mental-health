# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 10:12:20 2020

@author: GIM
"""
#loading all the required libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sn
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
data = pd.read_csv('C:\\Users\\GIM\\Downloads\\survey.csv')
data.head()
data.info()

#data cleaning starts

#dealing with missing data
#Unnecessary columns  
data = data.drop(['Timestamp'], axis= 1)
data = data.drop(['comments'], axis= 1)
data = data.drop(['state'], axis= 1)
data = data.drop(['no_employees'], axis= 1)
data = data.drop(['Country'], axis= 1)


missing_values = data.isnull().sum().sort_values(ascending = False)
data=data.fillna(0)

#Age having wrong values in negative. Keeping onlyvalues between 1 and 127
s = pd.Series(data['Age'])
s[s<18] = data['Age'].median()
data['Age'] = s
s = pd.Series(data['Age'])
s[s>120] = data['Age'].median()
data['Age'] = s


#Multiple values of gender for same group ex: male,m,male-ish all belonging to category Male. Clasified int 3 distinct groups of Male, Female and Trans
male_str = ["male", "m", "male-ish", "maile", "mal", "male (cis)", "make", "male ", "man","msle", "mail", "malr","cis man", "Cis Male", "cis male"]
trans_str = ["trans-female", "something kinda male?", "queer/she/they", "non-binary","nah", "all", "enby", "fluid", "genderqueer", "androgyne", "agender", "male leaning androgynous", "guy (-ish) ^_^", "trans woman", "neuter", "female (trans)", "queer", "ostensibly male, unsure what that really means"]          
female_str = ["cis female", "f", "female", "woman",  "femake", "female ","cis-female/femme", "female (cis)", "femail"]

for (row, col) in data.iterrows():

    if str.lower(col.Gender) in male_str:
        data['Gender'].replace(to_replace=col.Gender, value='male', inplace=True)

    if str.lower(col.Gender) in female_str:
        data['Gender'].replace(to_replace=col.Gender, value='female', inplace=True)

    if str.lower(col.Gender) in trans_str:
        data['Gender'].replace(to_replace=col.Gender, value='trans', inplace=True)



# Assign default values for each data type

defaultString = 'NaN'



data['self_employed'] = data['self_employed'].replace([defaultString], 'No')
print(data['self_employed'].unique())

data['care_options'] = data['care_options'].replace([defaultString], 'No')
print(data['care_options'].unique())
data['care_options'].unique()

data['wellness_program'] = data['wellness_program'].replace([defaultString], 'No')
print(data['wellness_program'].unique())
data['wellness_program'].unique()

#clean 'Gender'
#lower case all columm's elements
gender = data['Gender'].str.lower()
#print(gender)



#data cleaning ends

#correlation with treatment variable to find the top 10 highly correlated parameters


#Masking each categorical column
data['Gender1']=data.Gender.map({'female':0,'male':1,'trans':2})
data['self_employed1']=data.self_employed.map({'No':0,'Yes':1,})
data['family_history1']=data.family_history.map({'No':0,'Yes':1,})
data['treatment1']=data.treatment.map({'Yes':1,'No':0})
data['work_interfere1']=data.work_interfere.map({'Often':0,'Rarely':1,'Never':2,'Sometimes':3,'NaN':4})
data['remote_work1']=data.remote_work.map({'No':0,'Yes':1,})
data['tech_company1']=data.tech_company.map({'Yes':1,'No':0,})
data['benefits1']=data.benefits.map({'Yes':1, "Don't know":2, 'No':0})
data['care_options1']=data.care_options.map({'Not sure':2, 'No':0, 'Yes':1})
data['wellness_program1']=data.wellness_program.map({'No':0, "Don't know":2, 'Yes':1})
data['seek_help1']=data.seek_help.map({'Yes':1,"Don't know":2,'No':0})
data['anonymity1']=data.anonymity.map({'Yes':1,"Don't know":2,'No':0})
data['leave1']=data.leave.map({'Somewhat easy':0,"Don't know":1,'Somewhat difficult':2,
    'Very difficult':3,'Very easy':4})
data['mental_health_consequence1']=data.mental_health_consequence.map({'No':0,"Maybe":2,'Yes':1})
data['phys_health_consequence1']=data.phys_health_consequence.map({'No':0,"Yes":1,'Maybe':2})
data['coworkers1']=data.coworkers.map({'Some of them':2,"No":0,'Yes':1})
data['supervisor1']=data.supervisor.map({'Yes':1, 'No':0, 'Some of them':2})
data['mental_health_interview1']=data.mental_health_interview.map({'No':0, 'Yes':1, 'Maybe':2})
data['phys_health_interview1']=data.phys_health_interview.map({'Maybe':2, 'No':0, 'Yes':1})
data['mental_vs_physical1']=data.mental_vs_physical.map({'Yes':1, "Don't know":2, 'No':0})
data['obs_consequence1']=data.obs_consequence.map({'No':0, 'Yes':1})


#creating correlation matrix with heatmap



corrmat1 = data.corr()
print(corrmat1)

f, ax = plt.subplots(figsize=(12, 9))
sn.heatmap(corrmat1, vmax=.8, annot=True);
plt.show()

#treatment correlation matrix, selecting only the top 10 highly correlate parameters with treatment
k = 10 #number of variables for heatmap
cols = corrmat1.nlargest(k, 'treatment1')['treatment1'].index
cm = np.corrcoef(data[cols].values.T)
sn.set(font_scale=1.25)
hm = sn.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

#Outlier analysis using boxplot

#boxplot of all selected features

plt.boxplot(data['Age'])


#function written to remove outliers based on interquartile range
def remove_outlier(data, Age):
    q1 = data[Age].quantile(0.25)
    q3 = data[Age].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = data.loc[(data[Age] > fence_low) & (data[Age] < fence_high)]
    return df_out


#calling the function
data_leave = remove_outlier(data,'Age')
print(data_leave)



plt.boxplot(data_leave['Age'])



feature_cols = ['family_history1','obs_consequence1','mental_health_consequence1','Age','leave1','phys_health_consequence1','remote_work1','coworkers1','care_options1']
X = data_leave[feature_cols]
y = data_leave.treatment1

# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

 #BUILDING THE LOGIT MODEL
model_lr=LogisticRegression(solver='lbfgs')
model_lr.fit(X_train,y_train)


print('Coefficients of log r model')
coef=model_lr.coef_
intercept=model_lr.intercept_

#predicting train set to calculate accuracy of LR model
predicted_classes_lr=model_lr.predict(X_train)

print('Confusion matrix for lr model')
conf_mat_lr=confusion_matrix(y_train.tolist(),predicted_classes_lr)
print(conf_mat_lr)
sn.heatmap(conf_mat_lr,annot=True)
plt.xlabel('Predicted Classes')
plt.ylabel('Actual Classes')
plt.show()

#calculate accuracy scores
accuracy_lr=accuracy_score(y_train,predicted_classes_lr)
print('accuracy score (train) for lr model',accuracy_lr)

#SVM STARTING
svclassifier=SVC(kernel='linear')
svclassifier.fit(X_train,y_train)
predicted_classes_svm=svclassifier.predict(X_train)
accuracy_svm=accuracy_score(y_train,predicted_classes_svm)
print('accuracy score (train) for svm model',accuracy_svm)
print('Confusion matrix for svm model')
conf_mat_svm=confusion_matrix(y_train.tolist(),predicted_classes_svm)
print(conf_mat_svm)
sn.heatmap(conf_mat_svm,annot=True)
plt.xlabel('Predicted Classes')
plt.ylabel('Actual Classes')
plt.show()
print('confusion matrix (test set) for svm model')
predicted_test_classes_svm=svclassifier.predict(X_test)
conf_mat_test_svm=confusion_matrix(y_test.tolist(),predicted_test_classes_svm)
print(conf_mat_test_svm)
#svm ends

#knn
model_knn=KNeighborsClassifier()
model_knn.fit(X_train,y_train)
predicted_classes_knn=model_knn.predict(X_train)
accuracy_knn=accuracy_score(y_train,predicted_classes_knn)
print('accuracy score (train) for knn model',accuracy_knn)
print('Confusion matrix for knn model')
conf_mat_knn=confusion_matrix(y_train.tolist(),predicted_classes_knn)
print(conf_mat_knn)
sn.heatmap(conf_mat_knn,annot=True)
plt.xlabel('Predicted Classes')
plt.ylabel('Actual Classes')
plt.show()
print('confusion matrix (test set) for knn model')
predicted_test_classes_knn=model_knn.predict(X_test)
conf_mat_test_knn=confusion_matrix(y_test.tolist(),predicted_test_classes_knn)
print(conf_mat_test_knn)

#nb
model_nb=GaussianNB()
model_nb.fit(X_train,y_train)
predicted_classes_nb=model_nb.predict(X_train)
accuracy_nb=accuracy_score(y_train,predicted_classes_nb)
print('accuracy score (train) for nb model',accuracy_nb)
print('Confusion matrix for nb model')
conf_mat_nb=confusion_matrix(y_train.tolist(),predicted_classes_nb)
print(conf_mat_nb)
sn.heatmap(conf_mat_nb,annot=True)
plt.xlabel('Predicted Classes')
plt.ylabel('Actual Classes')
plt.show()
print('confusion matrix (test set) for knn model')
predicted_test_classes_nb=model_nb.predict(X_test)
conf_mat_test_nb=confusion_matrix(y_test.tolist(),predicted_test_classes_nb)
print(conf_mat_test_nb)



   
       
       
