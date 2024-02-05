#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install -U imbalanced-learn


# In[2]:


conda install -c glemaitre imbalanced-learn


# In[3]:


# data manipulation libraries
import numpy as np # linear algebra
import pandas as pd # data processing

# data visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# machine learning methods
from sklearn.model_selection import train_test_split # data splitting into train and test
from sklearn.preprocessing import FunctionTransformer
from imblearn.combine import SMOTETomek

from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_validate
from sklearn.metrics import f1_score


# In[4]:


dataset=pd.read_csv("Z:\\DOWNLOAD\\Documents\\Desktop\\7thsem\\ML\\dataset_sepsis\\MIMIC_DATASET.csv")


# In[5]:


dataset.isna().sum(axis=0).sort_values(ascending=False)/ len(dataset) * 100


# In[6]:


dataset.info()


# In[7]:


dataset.isna().sum(axis=0).sort_values(ascending=False)/ len(dataset) * 100


# In[8]:


dataset.nunique()


# In[9]:


print(dataset[dataset['Sepsis_label'] == 0].shape)

print(dataset[dataset['Sepsis_label'] == 1].shape)


# In[10]:


ratio_org = len(dataset[dataset['Sepsis_label'] == 1]) / len(dataset[dataset['Sepsis_label'] == 0])

print(ratio_org)


# In[11]:


dataset.drop('HADM_ID',axis=1,inplace=True)
# # dataset.drop('Patient_ID',axis=1,inplace=True)


# In[12]:


dataset.describe(include=('all'))


# In[13]:


dataset = dataset[(dataset['Age'] > 14) & (dataset['Age'] <= 90)]


# In[14]:


dataset['Sepsis_label'].value_counts()


# In[15]:


missing_feature_percentage_50=dataset.isna().sum(axis=1) / len(dataset.columns) * 100
rows_with_high_missing_features = missing_feature_percentage_50[missing_feature_percentage_50 >60].index.tolist()
print(len(rows_with_high_missing_features))
dataset=dataset.drop(rows_with_high_missing_features,axis=0)


# In[16]:


print(dataset[dataset['Sepsis_label'] == 0].shape)

print(dataset[dataset['Sepsis_label'] == 1].shape)


# In[17]:


missing_feature_percentage_50=dataset.isna().sum(axis=1) / len(dataset.columns) * 100
rows_with_high_missing_features = missing_feature_percentage_50[missing_feature_percentage_50 >60].index.tolist()
print(len(rows_with_high_missing_features))
dataset=dataset.drop(rows_with_high_missing_features,axis=0)


# In[18]:


dataset.isna().sum(axis=0).sort_values(ascending=False)/ len(dataset) * 100


# In[19]:


missing_row_percentage_70=dataset.isna().sum(axis=0) / len(dataset) * 100
missing_features = missing_row_percentage_70[missing_row_percentage_70 > 60].index.tolist()
print(len(missing_features))
dataset=dataset.drop(missing_features,axis=1)


# In[20]:


dataset.isna().sum(axis=0).sort_values(ascending=False)/ len(dataset) * 100


# In[21]:


# List of vital variables with their missing percentages
vital_variables = [
    "DBP",
    "Resp",
    "SBP",
    "O2Sat",
    "MAP",
    "HR",
]

# Threshold for the number of missing vital variables beyond which rows will be removed
threshold_missing_vitals = 4

# Count the number of missing vital variables for each row
missing_vital_counts = dataset[vital_variables].isna().sum(axis=1)

# Filter and keep only rows with missing vital counts less than or equal to the threshold
cleaned_dataset = dataset[missing_vital_counts <= threshold_missing_vitals]

# Display the cleaned dataset
# print(cleaned_dataset)
print(len(cleaned_dataset))


# In[22]:


dataset=cleaned_dataset
ratio_org = len(dataset[dataset['Sepsis_label'] == 1]) / len(dataset[dataset['Sepsis_label'] == 0])

print(ratio_org)


# In[23]:


dataset.describe(include=('all'))


# In[24]:


X =dataset.drop('Sepsis_label', axis=1)
y = dataset['Sepsis_label']


# In[25]:


dataset.isna().sum(axis=0).sort_values(ascending=False)/ len(dataset) * 100


# In[26]:


X_train,X_test,y_train, y_test = train_test_split(X,y,test_size=0.1,random_state=42)


# In[27]:


##        X   TRAIN
# reduced_dataset=dataset
# List of features to impute (exclude non-numeric and target features)
features_to_impute = [
    'INR','PTT', 'WBC','Hgb', 'WBC', 'Calcium', 'Hct', 'Magnesium', 'Creatinine',
    'HCO3', 'Chloride', 'BUN', 'Sodium', 'Potassium', 'Glucose', 'DBP', 'SBP', 'MAP','O2Sat',
    'Resp','HR'
]
# Create a subset of the dataset with only the features to impute
subset_data = X_train[features_to_impute].copy()

# Initialize the IterativeImputer
knn_imputer = KNNImputer(n_neighbors=10)

subset_data.shape


# In[ ]:





# In[28]:


##        X   TEST
subset_data1 = X_test[features_to_impute].copy()
subset_data1.shape


# In[29]:


#X train
# Fit and transform the imputer on the subset data
imputed_data = knn_imputer.fit_transform(subset_data)

# print(len(reduced_data))
# Replace the missing values in the original dataset with imputed values
# reduced_data[features_to_impute] = imputed_data

print(len(imputed_data))
# print(len(reduced_data))


# In[30]:


##        X   TEST
imputed_data1 = knn_imputer.transform(subset_data1)
# print(len(reduced_data))
print(len(imputed_data1))
# print(len(reduced_data))


# In[31]:


X_train[features_to_impute] = imputed_data

X_test[features_to_impute] = imputed_data1


# In[32]:


# # Get the top 15 feature names
# top_features = featureScores.nlargest(15, 'Score')['Specs']

# # Drop all other features from X_train except the top_features
# X_train = X_train[top_features]
# X_test = X_test[top_features]

X_train.describe()


# In[33]:


from imblearn.combine import SMOTETomek

smk=SMOTETomek(random_state=42)
X_train_res,y_train_res = smk.fit_resample(X_train,y_train)
X_test_res,y_test_res=smk.fit_resample(X_test,y_test)


# In[34]:


X_train_res.shape,y_train_res.shape, X_test_res.shape,y_test_res.shape


# In[35]:


y_train_res.value_counts()


# In[36]:


def evaluate_model(y_true,y_pred):
  accuracy = accuracy_score(y_true, y_pred)
  print("Accuracy:", accuracy)
  precision = precision_score(y_true, y_pred)
  print("Precision:", precision)
  recall = recall_score(y_true, y_pred)
  print("Recall:", recall)
  f1 = f1_score(y_true, y_pred)
  print("F1 Score:", f1)
#   auc = roc_auc_score(y_true, y_pred)
#   print("AUC-ROC:", auc)
#   mae = mean_absolute_error(y_true, y_pred)
#   print("Mean Absolute Error:", mae)
#   rmse = np.sqrt(mean_squared_error(y_true, y_pred))
#   print("Root Mean Squared Error:", rmse)
  cm = confusion_matrix(y_true, y_pred)
  sns.heatmap(cm, annot=True, fmt='d')
  plt.show()


# In[37]:


from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
skfold=StratifiedKFold(n_splits=5)
logreg = LogisticRegression(verbose=1,max_iter=2000)
logreg.fit(X_train_res, y_train_res)


scoring_metrics = ['f1', 'recall', 'precision', 'accuracy']
results = {}

# Iterate through the scoring metrics
for metric in scoring_metrics:
    scores = cross_val_score(logreg, X_train_res, y_train_res, cv=skfold, scoring=metric)
    results[metric] = scores

# Print the results
for metric, scores in results.items():
    print(f'Scores for {metric}: {scores}')
    print(f'Mean {metric}: {np.mean(scores):.2f}\n')
    print(f'Deviation {metric}: {2*np.std(scores):.2f}\n')
# cv_logreg = cross_validate(logreg, X_train_res, y_train_res, cv=10, scoring="f1", return_train_score=True)
# cv_logreg


# In[38]:


# logreg.fit(X_train_res, y_train_res)
lr_predictions = logreg.predict(X_test_res)
evaluate_model(y_test_res,lr_predictions)

from sklearn.metrics import mean_squared_error

y_train_pred = logreg.predict(X_train_res)  # Make predictions on the training data
training_error = mean_squared_error(y_train_res, y_train_pred)
print(training_error)

y_test_pred = logreg.predict(X_test_res)  # Make predictions on the testing data
testing_error = mean_squared_error(y_test_res, y_test_pred)
print(testing_error)


# In[39]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=7,metric='euclidean',weights='uniform')
knn.fit(X_train_res.values, y_train_res)
scoring_metrics = ['f1', 'recall', 'precision', 'accuracy']
results = {}

# Iterate through the scoring metrics
for metric in scoring_metrics:
    scores = cross_val_score(knn, X_train_res.values, y_train_res, cv=skfold, scoring=metric)
    results[metric] = scores

# Print the results
for metric, scores in results.items():
    print(f'Scores for {metric}: {scores}')
    print(f'Mean {metric}: {np.mean(scores):.2f}\n')
    print(f'Deviation {metric}: {2*np.std(scores):.2f}\n')
# cv_knn = cross_validate(knn, X_train_res.values, y_train_res, cv=10, scoring="f1", return_train_score=True)
# cv_knn


# In[40]:


knn.fit(X_train_res.values, y_train_res)
knn_predictions = knn.predict(X_test_res.values)

evaluate_model(y_test_res,knn_predictions)

y_train_pred = knn.predict(X_train_res.values)  # Make predictions on the training data
training_error = mean_squared_error(y_train_res, y_train_pred)
print(training_error)

y_test_pred = knn.predict(X_test_res.values)  # Make predictions on the testing data
testing_error = mean_squared_error(y_test_res, y_test_pred)
print(testing_error)


# In[41]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=10, verbose=1,max_features=10)
rf.fit(X_train_res, y_train_res)
scoring_metrics = ['f1', 'recall', 'precision', 'accuracy']
results = {}

# Iterate through the scoring metrics
for metric in scoring_metrics:
    scores = cross_val_score(rf, X_train_res, y_train_res, cv=skfold, scoring=metric)
    results[metric] = scores

# Print the results
for metric, scores in results.items():
    print(f'Scores for {metric}: {scores}')
    print(f'Mean {metric}: {np.mean(scores):.2f}\n')
    print(f'Deviation {metric}: {2*np.std(scores):.2f}\n')
# cv_rf = cross_validate(rf, X_train_res, y_train_res, cv=5, scoring="f1", return_train_score=True)
# cv_rf


# In[42]:


rf_predictions = rf.predict(X_test_res)
evaluate_model(y_test_res,rf_predictions)

y_train_pred = rf.predict(X_train_res)  # Make predictions on the training data
training_error = mean_squared_error(y_train_res, y_train_pred)
print(training_error)

y_test_pred = rf.predict(X_test_res)  # Make predictions on the testing data
testing_error = mean_squared_error(y_test_res, y_test_pred)
print(testing_error)


# In[43]:


from xgboost import XGBClassifier

xgboost = XGBClassifier(n_estimators=10,verbosity=1)
xgboost.fit(X_train_res, y_train_res)
scoring_metrics = ['f1', 'recall', 'precision', 'accuracy']
results = {}

# Iterate through the scoring metrics
for metric in scoring_metrics:
    scores = cross_val_score(xgboost, X_train_res, y_train_res, cv=skfold, scoring=metric)
    results[metric] = scores

# Print the results
for metric, scores in results.items():
    print(f'Scores for {metric}: {scores}')
    print(f'Mean {metric}: {np.mean(scores):.2f}\n')
    print(f'Deviation {metric}: {2*np.std(scores):.2f}\n')
# cv_xgboost = cross_validate(xgboost, X_train_res, y_train_res, cv=10, scoring="f1", return_train_score=True, verbose=1)
# cv_xgboost


# In[44]:


xgb_predictions = xgboost.predict(X_test_res)
evaluate_model(y_test_res,xgb_predictions)

y_train_pred = xgboost.predict(X_train_res)  # Make predictions on the training data
training_error = mean_squared_error(y_train_res, y_train_pred)
print(training_error)

y_test_pred = xgboost.predict(X_test_res)  # Make predictions on the testing data
testing_error = mean_squared_error(y_test_res, y_test_pred)
print(testing_error)


# In[45]:


from sklearn.ensemble import AdaBoostClassifier

adaboost = AdaBoostClassifier(n_estimators=10, random_state=5,algorithm='SAMME.R')
# adaboost.fit(X_train_res, y_train_res)

scoring_metrics = ['f1', 'recall', 'precision', 'accuracy']
results = {}

# Iterate through the scoring metrics
for metric in scoring_metrics:
    scores = cross_val_score(adaboost, X_train_res, y_train_res, cv=skfold, scoring=metric)
    results[metric] = scores

# Print the results
for metric, scores in results.items():
    print(f'Scores for {metric}: {scores}')
    print(f'Mean {metric}: {np.mean(scores):.2f}\n')
    print(f'Deviation {metric}: {2*np.std(scores):.2f}\n')
# cv_adaboost = cross_validate(adaboost, X_train_res, y_train_res, cv=10, scoring="f1", return_train_score=True)
# cv_adaboost


# In[46]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import numpy as np

adaboost = AdaBoostClassifier(n_estimators=10, random_state=5, algorithm='SAMME.R')

# Define your cross-validation strategy, e.g., StratifiedKFold
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring_metrics = ['f1', 'recall', 'precision', 'accuracy']
results = {}

for metric in scoring_metrics:
    scores = []

    for train_idx, val_idx in skfold.split(X_train_res, y_train_res):
        X_train_fold, X_val_fold = X_train_res.iloc[train_idx], X_train_res.iloc[val_idx]
        y_train_fold, y_val_fold = y_train_res.iloc[train_idx], y_train_res.iloc[val_idx]

        adaboost.fit(X_train_fold, y_train_fold)
        predictions = adaboost.predict(X_val_fold)

        if metric == 'f1':
            score = f1_score(y_val_fold, predictions)
        elif metric == 'recall':
            score = recall_score(y_val_fold, predictions)
        elif metric == 'precision':
            score = precision_score(y_val_fold, predictions)
        elif metric == 'accuracy':
            score = accuracy_score(y_val_fold, predictions)

        scores.append(score)

    results[metric] = scores

for metric, scores in results.items():
    mean_score = np.mean(scores)
    std_deviation = 2 * np.std(scores)
    print(f'Scores for {metric}: {scores}')
    print(f'Mean {metric}: {mean_score:.2f}')
    print(f'Deviation {metric}: {std_deviation:.2f}')

# Now, make predictions on the test set
ada_predictions = adaboost.predict(X_test_res)

# Evaluate the model on the test set using appropriate metrics (e.g., f1, recall, precision, accuracy)
evaluate_model(y_test_res, ada_predictions)


# In[47]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier


# In[48]:


from sklearn.neighbors import KNeighborsClassifier
tree = DecisionTreeClassifier(min_samples_leaf=4,max_depth=100)
knn = KNeighborsClasstree = DecisionTreeClassifier(min_samples_leaf=4,max_depth=100)
tree.fit(X_train_res, y_train_res)

scoring_metrics = ['f1', 'recall', 'precision', 'accuracy']
results = {}

# Iterate through the scoring metrics
for metric in scoring_metrics:
    scores = cross_val_score(tree, X_train_res, y_train_res, cv=skfold, scoring=metric)
    results[metric] = scores

# Print the results
for metric, scores in results.items():
    print(f'Scores for {metric}: {scores}')
    print(f'Mean {metric}: {np.mean(scores):.2f}\n')
    print(f'Deviation {metric}: {2*np.std(scores):.2f}\n')
# cv_tree = cross_validate(tree, X_train_res, y_train_res, cv=10, scoring="f1", return_train_score=True)
# cv_treeifier(n_neighbors=50)
knn.fit(X_train, y_train)


# In[49]:


tree = DecisionTreeClassifier(min_samples_leaf=4,max_depth=100)
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring_metrics = ['f1', 'recall', 'precision', 'accuracy']
results = {}

for metric in scoring_metrics:
    scores = []

    for train_idx, val_idx in skfold.split(X_train_res, y_train_res):
        X_train_fold, X_val_fold = X_train_res.iloc[train_idx], X_train_res.iloc[val_idx]
        y_train_fold, y_val_fold = y_train_res.iloc[train_idx], y_train_res.iloc[val_idx]

        tree.fit(X_train_fold, y_train_fold)
        predictions = tree.predict(X_val_fold)

        if metric == 'f1':
            score = f1_score(y_val_fold, predictions)
        elif metric == 'recall':
            score = recall_score(y_val_fold, predictions)
        elif metric == 'precision':
            score = precision_score(y_val_fold, predictions)
        elif metric == 'accuracy':
            score = accuracy_score(y_val_fold, predictions)

        scores.append(score)

    results[metric] = scores

for metric, scores in results.items():
    mean_score = np.mean(scores)
    std_deviation = 2 * np.std(scores)
    print(f'Scores for {metric}: {scores}')
    print(f'Mean {metric}: {mean_score:.2f}')
    print(f'Deviation {metric}: {std_deviation:.2f}')

# Now, make predictions on the test set
ada_predictions = tree.predict(X_test_res)

# Evaluate the model on the test set using appropriate metrics (e.g., f1, recall, precision, accuracy)
evaluate_model(y_test_res, ada_predictions)


# In[50]:


M1 = VotingClassifier(estimators=[('rf', rf), ('xgboost', xgboost)], voting='soft')
M1.fit(X_train_res, y_train_res)

scoring_metrics = ['f1', 'recall', 'precision', 'accuracy']
results = {}

# Iterate through the scoring metrics
for metric in scoring_metrics:
    scores = cross_val_score(M1, X_train_res, y_train_res, cv=skfold, scoring=metric)
    results[metric] = scores
#     # Evaluate the model on the test set
#     test_accuracy = M1.score(X_test_res, y_test_res)
#     print("Test Accuracy:", test_accuracy)

# Print the results
for metric, scores in results.items():
    print(f'Scores for {metric}: {scores}')
#     print(f'Test accuracy: {scores}')
    print(f'Mean {metric}: {np.mean(scores):.2f}\n')
    print(f'Deviation {metric}: {2*np.std(scores):.2f}\n')
# cv_ensemble = cross_validate(M1, X_train_res, y_train_res, cv=10, scoring="f1", return_train_score=True)
# cv_ensemble


# In[51]:


m1_predictions = M1.predict(X_test_res)
evaluate_model(y_test_res,m1_predictions)

y_train_pred = M1.predict(X_train_res)  # Make predictions on the training data
training_error = mean_squared_error(y_train_res, y_train_pred)
print(training_error)

y_test_pred = M1.predict(X_test_res)  # Make predictions on the testing data
testing_error = mean_squared_error(y_test_res, y_test_pred)
print(testing_error)


# In[52]:


M2 = VotingClassifier(estimators=[('rf', rf), ('xgboost', xgboost),('tree',tree)], voting='soft')
M2.fit(X_train_res, y_train_res)

scoring_metrics = ['f1', 'recall', 'precision', 'accuracy']
results = {}

# Iterate through the scoring metrics
for metric in scoring_metrics:
    scores = cross_val_score(M2, X_train_res, y_train_res, cv=skfold, scoring=metric)
    results[metric] = scores

# Print the results
for metric, scores in results.items():
    print(f'Scores for {metric}: {scores}')
    print(f'Mean {metric}: {np.mean(scores):.2f}\n')
    print(f'Deviation {metric}: {2*np.std(scores):.2f}\n')
# cv_ensemble = cross_validate(M2, X_train_res, y_train_res, cv=10, scoring="f1", return_train_score=True)
# cv_ensemble


# In[53]:


m2_predictions = M2.predict(X_test_res)
evaluate_model(y_test_res,m2_predictions)

y_train_pred = M2.predict(X_train_res)  # Make predictions on the training data
training_error = mean_squared_error(y_train_res, y_train_pred)
print(training_error)

y_test_pred = M2.predict(X_test_res)  # Make predictions on the testing data
testing_error = mean_squared_error(y_test_res, y_test_pred)
print(testing_error)


# In[54]:


from sklearn.metrics import roc_curve, roc_auc_score


# In[55]:


lr_scores = logreg.predict_proba(X_test_res)[:,1]
print("lr_scores",lr_scores)

knn_scores = knn.predict_proba(X_test_res.values)[:,1]
print("knn_scores",knn_scores)

rf_scores = rf.predict_proba(X_test_res)[:,1]
print("rf_scores",rf_scores)

xgb_scores = xgboost.predict_proba(X_test_res)[:,1]
print("xgb_scores",xgb_scores)

ada_scores = adaboost.predict_proba(X_test_res)[:,1]
print("ada_scores",ada_scores)

tree_scores = tree.predict_proba(X_test_res)[:,1]
print("tree_scores",tree_scores)

M1_scores = M1.predict_proba(X_test_res)[:,1]
print("M1_scores",M1_scores)

M2_scores = M2.predict_proba(X_test_res)[:,1]
print("M2_scores",M2_scores)

# svm_scores = svm.predict_proba(X_test_res)[:,1]
# print("xgb_scores",svm_scores)


# In[56]:


# Generate ROC curve data for logistic regression model
lr_fpr, lr_tpr, lr_thresholds = roc_curve(y_test_res, lr_scores)
print(lr_thresholds)
lr_auc = roc_auc_score(y_test_res, lr_scores)

# Generate ROC curve data for logistic regression model
knn_fpr, knn_tpr, knn_thresholds = roc_curve(y_test_res, knn_scores)
print(knn_thresholds)
knn_auc = roc_auc_score(y_test_res, knn_scores)

# Generate ROC curve data for logistic regression model
rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test_res, rf_scores)
print(rf_thresholds)
rf_auc = roc_auc_score(y_test_res, rf_scores)

# Generate ROC curve data for logistic regression model
xgb_fpr, xgb_tpr, xgb_thresholds = roc_curve(y_test_res, xgb_scores)
print(xgb_thresholds)
xgb_auc = roc_auc_score(y_test_res, xgb_scores)

# Generate ROC curve data for logistic regression model
ada_fpr, ada_tpr, ada_thresholds = roc_curve(y_test_res, ada_scores)
print(ada_thresholds)
ada_auc = roc_auc_score(y_test_res, ada_scores)

# Generate ROC curve data for logistic regression model
tree_fpr, tree_tpr, tree_thresholds = roc_curve(y_test_res, tree_scores)
print(tree_thresholds)
tree_auc = roc_auc_score(y_test_res, tree_scores)

# Generate ROC curve data for logistic regression model
M1_fpr, M1_tpr, M1_thresholds = roc_curve(y_test_res, M1_scores)
print(M1_thresholds)
M1_auc = roc_auc_score(y_test_res, M1_scores)

# Generate ROC curve data for logistic regression model
M2_fpr, M2_tpr, M2_thresholds = roc_curve(y_test_res, M2_scores)
print(M2_thresholds)
M2_auc = roc_auc_score(y_test_res, M2_scores)


# In[57]:


import plotly.graph_objects as go

# Generate a trace for the Logistic Regression ROC curve
trace0 = go.Scatter(
    x=lr_fpr,
    y=lr_tpr,
    mode='lines',
    name=f'Logistic Regression (Area = {lr_auc:.2f})'
)

# Generate a trace for the SVM ROC curve
trace1 = go.Scatter(
    x=knn_fpr,
    y=knn_tpr,
    mode='lines',
    name=f'knn (Area = {knn_auc:.2f})'
)

# Generate a trace for the rf ROC corve
trace2 = go.Scatter(
    x=rf_fpr,
    y=rf_tpr,
    mode='lines',
    name=f'Random Forest (Area = {rf_auc:.2f})'
)

# Generate a trace for the xgb ROC curve
trace3 = go.Scatter(
    x=xgb_fpr,
    y=xgb_tpr,
    mode='lines',
    name=f'XGB (Area = {xgb_auc:.2f})'
)

# Generate a trace for the xgb ROC curve
trace4 = go.Scatter(
    x=ada_fpr,
    y=ada_tpr,
    mode='lines',
    name=f'ADA (Area = {ada_auc:.2f})'
)

# Generate a trace for the xgb ROC curve
trace5 = go.Scatter(
    x=tree_fpr,
    y=tree_tpr,
    mode='lines',
    name=f'Tree (Area = {tree_auc:.2f})'
)

# Generate a trace for the xgb ROC curve
trace6 = go.Scatter(
    x=M1_fpr,
    y=M1_tpr,
    mode='lines',
    name=f'M1 (Area = {M1_auc:.2f})'
)

# Generate a trace for the xgb ROC curve
trace7 = go.Scatter(
    x=M2_fpr,
    y=M2_tpr,
    mode='lines',
    name=f'M2 (Area = {M2_auc:.2f})'
)

# Diagonal line
trace8 = go.Scatter(
    x=[0, 1], 
    y=[0, 1], 
    mode='lines', 
    name='Random (Area = 0.5)', 
    line=dict(dash='dash')
)



# In[58]:


data = [trace0, trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8]

# Define layout with square aspect ratio
layout = go.Layout(
    title='Receiver Operating Characteristic',
    xaxis=dict(title='False Positive Rate'),
    yaxis=dict(title='True Positive Rate'),
    autosize=False,
    width=800,
    height=800,
    showlegend=True
)

# Define figure and add data
fig = go.Figure(data=data, layout=layout)

# Show figure
fig.show()


# In[71]:





# In[75]:





# In[72]:





# In[ ]:




