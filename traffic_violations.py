#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_score,recall_score, confusion_matrix, classification_report,accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer

from sklearn.utils.validation import check_is_fitted
import plotly.express as px

# Custom written utilities for analysis/modelling/feature engineering
from utils import FeatureImportance, pre_process, get_scores
import pickle
import plotly.io as pio
pio.renderers.default = "svg"


# ## Load data downloaded from S3

# In[4]:


# Run this incase data is not loaded via bash script
# !wget https://s3-us-west-2.amazonaws.com/pcadsassessment/parking_citations.corrupted.csv


# In[5]:

print(f"Loading Data #############")

df = pd.read_csv("parking_citations.corrupted.csv")

# In[8]:


# Finding Top 25 Make based on number of rows
pd_df = df.groupby("Make")['Ticket number'].count().reset_index(name="count").sort_values("count",ascending=False)[:25]
pd_df.head()


# In[9]:


# Creating the binary variable indicating whether popular Make or not
popularMakes = pd_df["Make"].values.tolist()
df['popularMakeorNot'] = df['Make'].apply(lambda x: 1 if x in popularMakes else 0,1)
df


# In[10]:


# Based on task description, separating corrupted rows (having NA values for Make column) as test and rest as training set
df_train = df[~df.Make.isna()]
df_test = df[df.Make.isna()]


# In[11]:

del df


# ## Feature Engineering + Modelling

# In[29]:


# Primary Features cps
feats = ['Issue Date', 'Issue time',
       'RP State Plate', 'Body Style',
       'Color', 'Agency', 'Violation code',
         "Route",'Fine amount', 'Latitude', 'Longitude']
out = ["popularMakeorNot"]


# In[30]:

print(f"Splitting into Train-Test Data #############")
X_train, X_test, y_train, y_test = train_test_split(df_train[feats], df_train[out], test_size=0.33, random_state=42,stratify=df_train[out])


# In[31]:


# Categorical features
cat_cols = ['RP State Plate', 'Body Style', 'Color', 'Agency', 'Violation code',"Route",
        'IssueHour', 'IssueWeek', 'IssueYear', 'IssueWeekDay', 'IssueDay']


# In[33]:

print(f"Pre-processing + Feature engineering from Data #############")

X_trainN = pre_process(X_train,cat_cols)
X_testN = pre_process(X_test,cat_cols)


# In[34]:


missing_num_vars = ["Latitude","Longitude","Fine amount"]
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

categorical_transformer =Pipeline(
    steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("one-hot encoder", 
 OneHotEncoder(handle_unknown="ignore"))]
)


preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, missing_num_vars),
        ("cat", categorical_transformer, cat_cols),
    ],remainder='passthrough'
)


# In[35]:


clf = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", RandomForestClassifier(random_state=42,max_depth=2,max_features=8,n_jobs=4,verbose=1))]
)


# In[36]:

print(f"Training Model #############")
clf.fit(X_trainN,y_train.values)


# In[37]:



y_pred = clf.predict(X_trainN)
print(f"\\n ######## Train set metrics ###### \\n")
get_scores(y_train.values,y_pred)

y_pred = clf.predict(X_testN)
print(f"\\n ######## Test set metrics ###### \\n")
get_scores(y_test.values,y_pred)


# ## Model Explanations + Strength/Weakness + Feasability 

# In[38]:


# feature_importance = FeatureImportance(clf)
# feature_importance.plot(top_n_features=50)


# It can be seen from feature importance how certain violation codes, body style and routes play a major role in determining whether a Make is popular or not whic makes sense inituitevly as well.
# 
# Also model accuracy+precision,recall metrics for training and test are almost same which indicates that there's not much overfitting in the model. Since we are using basic Random Forest model it can be deployed easily using the pickle file and nothing complicated.
# 

# We can say that for the task of predicitng popular make for corrupted data it should work pretty great since it works well for unseen test split we created. Furthermore, it has high precision as well as recall given we have high F-score.
# Although we would also say since data is imbalanced (1->90% and 0->10%) it is relatively easy to predict that make is popular so error should be low by default.

# In[39]:

print(f"Saving Model #############")
# Save model for making prediciton using server.py
pickle.dump(clf, open('model.pkl', 'wb'))

