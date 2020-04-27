from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,f1_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import graphviz 
from sklearn import tree

#access the csv files
train_df = pd.read_csv('train.csv')
train_df.info()
test_df = pd.read_csv('test.csv')

train_df = train_df.drop(columns=['Loan_ID']) 
test_df = test_df.drop(columns=['Loan_ID'])

categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Loan_Amount_Term', 'Property_Area','Credit_History']
numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']

#plotting the data from train.csv on graphs
fig,axes = plt.subplots(4,2,figsize=(12,15))
for idx,cat_col in enumerate(categorical_columns):
    row,col = idx//2,idx%2
    sns.countplot(x=cat_col,data=train_df,hue='Loan_Status',ax=axes[row,col])

fig.savefig('categorical_columns.png')

plt.subplots_adjust(hspace=1)

fig,axes = plt.subplots(1,3,figsize=(17,5))
for idx,cat_col in enumerate(numerical_columns):
    sns.boxplot(y=cat_col,data=train_df,x='Loan_Status',ax=axes[idx])

fig.savefig('numerical_columns.png')

print(train_df[numerical_columns].describe())
plt.subplots_adjust(hspace=1)


#turning the data into a format that is more readable for a computer
train_df_encoded = pd.get_dummies(train_df,drop_first=True)
train_df_encoded.head()
test_df_encoded = pd.get_dummies(test_df,drop_first=True)
test_df_encoded.head()

x = train_df_encoded.drop(columns='Loan_Status_Y')
y = train_df_encoded['Loan_Status_Y']

x_train = x
y_train = y
x_test = test_df_encoded


imp = SimpleImputer(strategy='mean')
imp_train = imp.fit(x_train)
x_train = imp_train.transform(x_train)
x_test_imp = imp_train.transform(x_test)

#training our classifier based off of the list x_train which was taken from train.csv
tree_clf = DecisionTreeClassifier()
tree_clf.fit(x_train,y_train)
y_pred = tree_clf.predict(x_train)
print("Training Data Set Accuracy: ", accuracy_score(y_train,y_pred))
print("Training Data F1 Score ", f1_score(y_train,y_pred))

print("Validation Mean F1 Score: ",cross_val_score(tree_clf,x_train,y_train,cv=5,scoring='f1_macro').mean())
print("Validation Mean Accuracy: ",cross_val_score(tree_clf,x_train,y_train,cv=5,scoring='accuracy').mean())

#graphing the accuracy of the predictions
training_accuracy = []
val_accuracy = []
training_f1 = []
val_f1 = []
min_samples_leaf = []
import numpy as np
for samples_leaf in range(1,80,3):
    tree_clf = DecisionTreeClassifier(max_depth=3,min_samples_leaf = samples_leaf)
    tree_clf.fit(x_train,y_train)
    y_training_pred = tree_clf.predict(x_train)

    training_acc = accuracy_score(y_train,y_training_pred)
    train_f1 = f1_score(y_train,y_training_pred)
    val_mean_f1 = cross_val_score(tree_clf,x_train,y_train,cv=5,scoring='f1_macro').mean()
    val_mean_accuracy = cross_val_score(tree_clf,x_train,y_train,cv=5,scoring='accuracy').mean()
    
    training_accuracy.append(training_acc)
    val_accuracy.append(val_mean_accuracy)
    training_f1.append(train_f1)
    val_f1.append(val_mean_f1)
    min_samples_leaf.append(samples_leaf)
    

Tuning_min_samples_leaf = {"Training Accuracy": training_accuracy, "Validation Accuracy": val_accuracy, "Training F1": training_f1, "Validation F1":val_f1, "Min_Samples_leaf": min_samples_leaf }
Tuning_min_samples_leaf_df = pd.DataFrame.from_dict(Tuning_min_samples_leaf)

plot_df = Tuning_min_samples_leaf_df.melt('Min_Samples_leaf',var_name='Metrics',value_name="Values")
fig,ax = plt.subplots(figsize=(15,5))
sns.pointplot(x="Min_Samples_leaf", y="Values",hue="Metrics", data=plot_df,ax=ax)

fig.savefig('graph.png')

#making predictions for the data from test.csv
from sklearn.metrics import confusion_matrix
tree_clf = DecisionTreeClassifier(max_depth=3,min_samples_leaf = 35)
tree_clf.fit(x_train,y_train)
y_pred = tree_clf.predict(x_test_imp)
y = y_pred

#putting the dataframe of test.csv into testPredictions.csv along with the Loan_Status column, with all of the predictions from our classifier
from csv import writer
from csv import reader
with open('testPredictions.csv', 'w', newline='') as write_obj, open('test.csv', 'r') as reader_obj:
  csv_writer = writer(write_obj)
  csv_reader = reader(reader_obj)
  i = 0
  for row in csv_reader:
    if i == 0:
      row.append('Loan_Status')
      csv_writer.writerow(row)
    else:
      if y[i-1] == 1:
        row.append('Y')
        csv_writer.writerow(row)
      else:
        row.append('N')
        csv_writer.writerow(row)
    i += 1

