#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


FifaMalePlayers = pd.read_csv("C:\\Users\\user\\Desktop\\AI\\male_players (legacy).csv")


# In[3]:


Players22 = pd.read_csv("C:\\Users\\user\\Desktop\\AI\\players_22-1.csv")


# In[4]:


FifaMalePlayers.head()


# In[5]:


#dropping unnecessary columns
FifaMalePlayers.drop('player_id', axis=1, inplace=True)
FifaMalePlayers.drop('player_url', axis=1, inplace=True)
FifaMalePlayers.drop('fifa_version', axis=1, inplace=True)
FifaMalePlayers.drop('fifa_update', axis=1, inplace=True)
FifaMalePlayers.drop('fifa_update_date', axis=1, inplace=True)
FifaMalePlayers.drop('short_name', axis=1, inplace=True)
FifaMalePlayers.drop('long_name', axis=1, inplace=True)
FifaMalePlayers.drop('player_positions', axis=1, inplace=True)
FifaMalePlayers.drop('player_face_url', axis=1, inplace=True)
FifaMalePlayers.drop('dob', axis=1, inplace=True)
FifaMalePlayers.drop('league_name', axis=1, inplace=True)
FifaMalePlayers.drop('nationality_name', axis=1, inplace=True)
FifaMalePlayers.drop('real_face', axis=1, inplace=True)


# In[6]:


FifaMalePlayers.drop('club_name', axis=1, inplace=True)


# In[7]:


FifaMalePlayers.shape


# In[8]:


FifaMalePlayers.head()


# In[9]:


#Removing columns with more than 30% missing values
V_more = []
V_less = []
for i in FifaMalePlayers.columns:
    if((FifaMalePlayers[i].isnull().sum())<(0.4*(FifaMalePlayers.shape[0]))):
        V_more.append(i)
    else:
        V_less.append(i)


# In[10]:


FifaMalePlayers = FifaMalePlayers[V_more]


# In[11]:


FifaMalePlayers.head()


# In[12]:


#Separating numeric and non-numeric data
import numpy as np
numeric_data = FifaMalePlayers.select_dtypes(include=np.number)
non_numeric_data = FifaMalePlayers.select_dtypes(include=['object'])


# In[13]:


#Missing values imputation
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imp = IterativeImputer(max_iter=10, random_state=0)
numeric_data = pd.DataFrame(np.round(imp.fit_transform(numeric_data)), columns=numeric_data.columns)


# In[14]:


non_numeric_data.columns


# In[15]:


numeric_data['league_level'].isnull().sum()


# In[16]:


df_train = pd.concat([numeric_data, non_numeric_data], axis=1)


# In[111]:


df_train.head()


# In[18]:


#Refining non_numeric data
columns_to_keep = ['preferred_foot', 'work_rate']
column_to_drop = [col for col in non_numeric_data if col not in columns_to_keep]


# In[19]:


non_numeric_data.drop(column_to_drop, axis=1, inplace=True)


# In[20]:


non_numeric_data


# In[21]:


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

non_numeric_data['preferred_foot'] = label_encoder.fit_transform(non_numeric_data['preferred_foot'])
non_numeric_data['work_rate'] = label_encoder.fit_transform(non_numeric_data['work_rate'])


# In[22]:


non_numeric_data.iloc[125:181,:]


# In[23]:


df_train = pd.concat([numeric_data, non_numeric_data], axis=1)


# In[24]:


df_train.head()


# In[25]:


#calculating correlation betweeen overall rating as an independent variable and other variables
corr_matrix = df_train.corr() 
correlation = corr_matrix['overall'].sort_values(ascending=False)


# In[26]:


correlation


# In[27]:


#Selecting variables with highest correlation with overall rating
relevant_variables = correlation[abs(correlation)>0.1].index.tolist()


# In[28]:


relevant_variables


# In[29]:


#Renaming the dataframe
Df_train = relevant_variables
relevant_variables = Df_train


# In[30]:


relevant_variables


# In[31]:


#creating feature subsets
X = df_train[relevant_variables]
y = df_train['overall']


# In[32]:


#Creating and training a suitable machine learning model using cross validation


# In[33]:


from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pickle as pkl


# In[34]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# In[35]:


#Scaling data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.fit_transform(X_val)


# In[36]:


#model declaration
models = {
    'Random Forest': RandomForestRegressor(random_state=42),
    'Linear Regression': LinearRegression(),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'AdaBoost': AdaBoostRegressor(random_state=42)
}


# In[37]:


#creating a dictionary to hold all the models' results
models_results = {}


# In[38]:


k_fold = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    #cross validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=k_fold, scoring='neg_mean_squared_error')
    
    #average score in all folds while converting all negative mse to positive
    average_mse = -cv_scores.mean()
    
    #fitting the model to the entire dataset
    model.fit(X_train, y_train)
    
    #saving the models using pickle
    with open("C:\\Users\\user\\Desktop\\AI\\male_players (legacy).csv" + name + '.pkl', 'wb') as f:
        pkl.dump(model, f)
    
    #storing all the models results and mse
    models_results[name] = {'model': model, 'average_cv_scores': average_mse}


# In[39]:


for name, result in models_results.items():
    print(name)
    print(average_mse)


# In[40]:


#Measuring models' performance and fine-tune


# In[41]:


from sklearn.metrics import r2_score
#creating a dictionary to hold other metrics
metrics = {}

for name, model in models_results.items():
    model = result['model']
    
    #predict on validation set
    y_pred = model.predict(X_val)
    
    #calculating mse
    mse = mean_squared_error(y_val, y_pred)
    
    #calcutating R_squared score
    r2 = r2_score(y_val, y_pred)
    
    #store metrics
    metrics[name] = {'mse': mse, 'r2':r2}
    
    print(name)
    print(mse)
    print(r2)


# In[42]:


#fine-tuning and optimization of models using RandomizedSearchCV 


# In[43]:


#Random forest


# In[48]:


from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV


# In[50]:


rf = RandomForestRegressor(random_state=42)

parameters = {
    'n_estimators':randint(100,200),
    'max_depth': [None, 10, 20],
    'min_samples_split': randint(2,10),
    'min_samples_leaf': randint(1,4)
}

#RandomizedSearchCV
random_sr_rf = RandomizedSearchCV(estimator=rf, param_distributions=parameters, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
random_sr_rf.fit(X_train, y_train)

#best params and best scores
random_sr_rf.best_params_
-random_sr_rf.best_score_

#updating model results dictionary
best_rf_model = random_sr_rf.best_estimator_
models_results['Random Forest']['model'] = best_rf_model


# In[51]:


#Gradient Boosting


# In[52]:


gb = GradientBoostingRegressor(random_state=42)

parameters_gb = {
    'n_estimators':randint(100,200),
    'max_depth': [None, 10, 20],
    'min_samples_split': randint(2,10),
    'min_samples_leaf': randint(1,4)
}

#RandomizedSearchCV
random_sr_gb = RandomizedSearchCV(estimator=gb, param_distributions=parameters_gb, cv=k_fold, scoring='neg_mean_squared_error', n_jobs=-1)
random_sr_gb.fit(X_train, y_train)

#best params and best scores
random_sr_gb.best_params_
-random_sr_gb.best_score_
 
#updating model results dictionary
best_gb_model = random_sr_gb.best_estimator_
models_results['Gradient Boosting']['model'] = best_gb_model


# In[53]:


#Ada Boost
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, VotingRegressor


# In[59]:


from scipy.stats import uniform
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold
Ada = AdaBoostRegressor(random_state=42)

parameters_ada = {
    'n_estimators':randint(100,200),
    'learning_rate': uniform(0.01, 1.0),
    'loss': ['linear', 'square', 'exponential']
    
}

#RandomizedSearchCV
random_sr_ada = RandomizedSearchCV(estimator=Ada, param_distributions=parameters_ada, cv=k_fold, scoring='neg_mean_squared_error', n_jobs=-1)
random_sr_ada.fit(X_train, y_train)

#best params and best scores
random_sr_ada.best_params_
-random_sr_ada.best_score_

#updating model results dictionary
best_ada_model = random_sr_ada.best_estimator_
models_results['AdaBoost']['model'] = best_ada_model


# In[60]:


#Ensemble model


# In[61]:


import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, VotingRegressor
import pickle as pkl
from sklearn.metrics import mean_squared_error


# In[69]:


#load data for all the models
with open('C:\\Users\\user\\Desktop\\AI\\male_players (legacy).csvRandom Forest.pkl','rb') as f:
    rf_model = pkl.load(f)

with open('C:\\Users\\user\\Desktop\\AI\\male_players (legacy).csvGradient Boosting.pkl', 'rb') as f:
    gb_model = pkl.load(f)

    
with open('C:\\Users\\user\\Desktop\\AI\\male_players (legacy).csvAdaBoost.pkl', 'rb') as f:
    ada_model = pkl.load(f)


# In[70]:


#create the ensemble model using voting regressor


# In[ ]:





# In[72]:


#Test how good is a model


# In[113]:


Players22.head()


# In[105]:


#dropping unnecessary columns
Players22.drop('sofifa_id', axis=1, inplace=True)
Players22.drop('fifa_version', axis=1, inplace=True)
Players22.drop('fifa_update', axis=1, inplace=True)
Players22.drop('fifa_update_date', axis=1, inplace=True)
Players22.drop('player_positions', axis=1, inplace=True)
Players22.drop('player_face_url', axis=1, inplace=True)
Players22.drop('dob', axis=1, inplace=True)
Players22.drop('league_name', axis=1, inplace=True)
Players22.drop('nationality_name', axis=1, inplace=True)
Players22.drop('real_face', axis=1, inplace=True)
Players22.drop('club_logo_url', axis=1, inplace=True)
Players22.drop('nation_logo_url', axis=1, inplace=True)
Players22.drop('nation_flag_url', axis=1, inplace=True)
Players22.drop('short_name', axis=1, inplace=True)
Players22.drop('long_name', axis=1, inplace=True)
Players22.drop('club_name', axis=1, inplace=True)
Players22.drop('club_flag_url', axis=1, inplace=True)
Players22.drop('club_position', axis=1, inplace=True)
Players22.drop('club_team_id', axis=1, inplace=True)
Players22.drop('club_contract_valid_until', axis=1, inplace=True)
Players22.drop('nationality_id', axis=1, inplace=True)


# In[161]:


Players22.drop('league_level', axis=1, inplace=True)
Players22.drop('club_jersey_number', axis=1, inplace=True)


# In[162]:


#Removing columns with more than 30% missing values
T_more = []
T_less = []
for i in Players22.columns:
    if((Players22[i].isnull().sum())<(0.4*(Players22.shape[0]))):
        T_more.append(i)
    else:
        T_less.append(i)

Players22 = Players22[T_more]


# In[163]:


#Separating numeric and non-numeric data
import numpy as np
T_numeric_data = Players22.select_dtypes(include=np.number)
T_non_numeric_data = Players22.select_dtypes(include=['object'])


# In[164]:


#Missing values imputation
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imp = IterativeImputer(max_iter=10, random_state=0)
T_numeric_data = pd.DataFrame(np.round(imp.fit_transform(T_numeric_data)), columns=T_numeric_data.columns)


# In[179]:


T_numeric_data.head()


# In[180]:


#Refining non_numeric data
columns_to_keep = ['preferred_foot', 'work_rate']
column_to_drop = [col for col in non_numeric_data if col not in columns_to_keep]


T_non_numeric_data.drop(column_to_drop, axis=1, inplace=True)

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

T_non_numeric_data['preferred_foot'] = label_encoder.fit_transform(T_non_numeric_data['preferred_foot'])
T_non_numeric_data['work_rate'] = label_encoder.fit_transform(T_non_numeric_data['work_rate'])


# In[181]:


df_test = pd.concat([T_numeric_data, T_non_numeric_data], axis=1)


# In[182]:


df_test


# In[ ]:





# In[206]:


missing_columns = [col for col in df_train.columns if col not in df_test.columns]

#adding missing columns with default values (e.g., 0)
for col in missing_columns:
    df_test[col] = 0

#ensuring the testing dataset has the same order of columns as the training dataset
X_df_test = df_test[df_train.columns]

#load it and transform the testing data
scaler = StandardScaler()
X_df_test = scaler.fit_transform(X_df_test)


# In[184]:


base_models = [rf_model, gb_model, ada_model]


# In[185]:


def ensemble_predict(models, X):
    predictions = np.zeros(len(X))
    for model in models:
        predictions += model.predict(X)
    predictions /= len(models)
    return predictions


# In[186]:


y_pred_ensemble = ensemble_predict(base_models, X_val)


# In[197]:


df_test


# In[203]:


all_zero_columns = df_test.columns[(df_test == 0).all()]
df_cleaned = df_test.drop(all_zero_columns, axis=1)


# In[204]:


df_cleaned.


# In[187]:


ensemble_rmse = np.sqrt(mean_squared_error(y_val, y_pred_ensemble))


# In[188]:


ensemble_rmse


# In[189]:


import joblib
joblib.dump(base_models, 'ensemble_model.pkl')


# In[190]:


ensemble_model = joblib.load('ensemble_model.pkl')


# In[208]:


y_pred_ensemble = ensemble_predict(ensemble_model, df_test)


# In[86]:


ensemble_model = VotingRegressor(estimators=[])


# In[212]:


#instantiating the voting regressor
vot_regressor = VotingRegressor(estimators=ensemble_model)

#predict based based on Players22 dataset
y_pred_em = vot_regressor.predict(X_df_test)


# In[213]:


#evaluating model's performance
mse_em = mean_squared_error(y_true_rating, y_pred_em)


# In[214]:


print('Random Forest Model Performance on players_22:',mse_em)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




