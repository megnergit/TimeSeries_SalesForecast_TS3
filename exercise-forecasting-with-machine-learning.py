#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# **This notebook is an exercise in the [Time Series](https://www.kaggle.com/learn/time-series) course.  You can reference the tutorial at [this link](https://www.kaggle.com/ryanholbrook/forecasting-with-machine-learning).**
# 
# ---
# 

# # Introduction #
# 
# Run this cell to set everything up!

# In[1]:


# Setup feedback system
from learntools.core import binder
binder.bind(globals())
from learntools.time_series.ex6 import *

# Setup notebook
from pathlib import Path
import ipywidgets as widgets
from learntools.time_series.style import *  # plot style settings
from learntools.time_series.utils import (create_multistep_example,
                                          load_multistep_data,
                                          make_lags,
                                          make_multistep_target,
                                          plot_multistep)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import RegressorChain
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor


comp_dir = Path('../input/store-sales-time-series-forecasting')

store_sales = pd.read_csv(
    comp_dir / 'train.csv',
    usecols=['store_nbr', 'family', 'date', 'sales', 'onpromotion'],
    dtype={
        'store_nbr': 'category',
        'family': 'category',
        'sales': 'float32',
        'onpromotion': 'uint32',
    },
    parse_dates=['date'],
    infer_datetime_format=True,
)
store_sales['date'] = store_sales.date.dt.to_period('D')
store_sales = store_sales.set_index(['store_nbr', 'family', 'date']).sort_index()

family_sales = (
    store_sales
    .groupby(['family', 'date'])
    .mean()
    .unstack('family')
    .loc['2017']
)

test = pd.read_csv(
    comp_dir / 'test.csv',
    dtype={
        'store_nbr': 'category',
        'family': 'category',
        'onpromotion': 'uint32',
    },
    parse_dates=['date'],
    infer_datetime_format=True,
)
test['date'] = test.date.dt.to_period('D')
test = test.set_index(['store_nbr', 'family', 'date']).sort_index()


# -------------------------------------------------------------------------------
# 
# Consider the following three forecasting tasks:
# 
# a. 3-step forecast using 4 lag features with a 2-step lead time<br>
# b. 1-step forecast using 3 lag features with a 1-step lead time<br>
# c. 3-step forecast using 4 lag features with a 1-step lead time<br>
# 
# Run the next cell to see three datasets, each representing one of the tasks above.

# In[2]:


datasets = load_multistep_data()

data_tabs = widgets.Tab([widgets.Output() for _ in enumerate(datasets)])
for i, df in enumerate(datasets):
    data_tabs.set_title(i, f'Dataset {i+1}')
    with data_tabs.children[i]:
        display(df)

display(data_tabs)


# # 1) Match description to dataset
# 
# Can you match each task to the appropriate dataset?

# In[3]:


# YOUR CODE HERE: Match the task to the dataset. Answer 1, 2, or 3.
task_a = 2
task_b = 1
task_c = 3

# Check your answer
q_1.check()


# In[4]:


# Lines below will give you a hint or solution code
#q_1.hint()
#q_1.solution()


# In[5]:


test_org = test.copy()
display(test_org.head(3))


# -------------------------------------------------------------------------------
# 
# Look at the time indexes of the training and test sets. From this information, can you identify the forecasting task for *Store Sales*?

# In[6]:


print("Training Data", "\n" + "-" * 13 + "\n", store_sales)
print("\n")
print("Test Data", "\n" + "-" * 9 + "\n", test)


# # 2) Identify the forecasting task for *Store Sales* competition
# 
# Try to identify the *forecast origin* and the *forecast horizon*. How many steps are within the forecast horizon? What is the lead time for the forecast?
# 
# Run this cell after you've thought about your answer.

# In[7]:


# View the solution (Run this cell to receive credit!)
q_2.check()


# In[8]:


from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import RegressorChain
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

import plotly.graph_objs as go
from plotly.subplots import make_subplots


DATA_DIR = Path('../input/store-sales-time-series-forecasting')

# mainly dtype object => category or int => category 
store_sales = pd.read_csv(DATA_DIR/'train.csv', 
                          dtype={
                              'store_nbr':'category',
                              'family':'category'
                          },
                          parse_dates=['date'], 
                          index_col='date')

store_sales.index = store_sales.index.to_period('D')
store_sales.drop(['id'], errors='ignore', axis=1, inplace=True)

# in oder to group with family, family has to be in column
store_sales.set_index(['store_nbr', 'family'], append=True, inplace=True)
store_sales = store_sales.sort_index()

family_sales = store_sales.groupby(['date', 'family']).mean().unstack().loc['2017']
# print(store_sales.info())
# print(store_sales.index)
# display(store_sales.head(3))
# display(family_sales.head(3))

test = pd.read_csv(DATA_DIR/'test.csv', 
                          dtype={
                              'store_nbr':'category',
                              'family':'category'
                          },
                          parse_dates=['date'])

test['date'] = test['date'].dt.to_period('D')
test = test.sort_values(['store_nbr', 'family'])
test = test[['store_nbr','family', 'date', 'id', 'onpromotion']]
test.set_index(['store_nbr', 'family', 'date'], inplace=True)
test.head(3)
# test.set_index(['store_nbr', 'family', 'date'], append=True, inplace=True)
# test = test.sort_index(1)
display(test.head(3))
print(family_sales.index.min(), family_sales.index.max())
print(test.index.get_level_values(2).min(), test.index.get_level_values(2).max())


# -------------------------------------------------------------------------------
# 
# In the tutorial we saw how to create a multistep dataset for a single time series. Fortunately, we can use exactly the same procedure for datasets of multiple series.
# 
# # 3) Create multistep dataset for *Store Sales*
# 
# Create targets suitable for the *Store Sales* forecasting task. Use 4 days of lag features. Drop any missing values from both targets and features.

# In[9]:


# {f'y_lag_{i}': y.shift(i) for i in range(1, n_lag)}


# In[10]:


y = family_sales.loc[:,'sales'] 
n_lag = 4
X = pd.concat({f'y_lag_{i}': y.shift(i) for i in range(1, (n_lag+1))}, axis=1).dropna()
# X.head(3)
y.head(3)
n_step = 16
y = pd.concat({f'y_step_{i+1}': y.shift(-i) for i in range(n_step)}, axis=1).dropna()
y, X = y.align(X, join='inner', axis=0)
q_3.check()


# In[ ]:





# In[11]:


# family_sales_org = family_sales.copy()
# len(family_sales_org)


# In[12]:


# YOUR CODE HERE
y = family_sales.loc[:, 'sales']

# YOUR CODE HERE: Make 4 lag features
# X = ____
n_lag =4
X = pd.concat({f'y_lag_{i}': y.shift(i) for i in range(1, n_lag+1)}, axis=1).dropna()

# YOUR CODE HERE: Make multistep target
# y = ____
n_step = 16
y = pd.concat({f'y_step_{i+1}': y.shift(-i) for i in range(n_step)}, axis=1).dropna()
y, X = y.align(X, join='inner', axis=0)

# Check your answer
q_3.check()


# In[13]:


# display(family_sales_org.head(3))
# display(family_sales.head(3))
# print(family_sales_org.info())
# print(family_sales.info())


# In[14]:


def make_lags(ts, lags, lead_time=1):
    return pd.concat(
        {
            f'y_lag_{i}': ts.shift(i)
            for i in range(lead_time, lags + lead_time)
        },
        axis=1)


# In[15]:


# y = family_sales_org.loc[:, 'sales']
# X = make_lags(y, lags=4).dropna()
# len(X)


# In[16]:


len(X)


# In[17]:


# y = family_sales_org.loc[:, 'sales']
# X = make_lags(y, lags=4).dropna()
# y = make_multistep_target(y, steps=16).dropna()
# # y, X = y.align(X, join='inner', axis=0)
# q_3.check()


# In[18]:


y.head()


# In[ ]:





# In[19]:


# Lines below will give you a hint or solution code
q_3.hint()
q_3.solution()


# -------------------------------------------------------------------------------
# 
# In the tutorial, we saw how to forecast with the MultiOutput and Direct strategies on the *Flu Trends* series. Now, you'll apply the DirRec strategy to the multiple time series of *Store Sales*.
# 
# Make sure you've successfully completed the previous exercise and then run this cell to prepare the data for XGBoost.

# In[20]:


le = LabelEncoder()
X1 = X.stack('family').reset_index('family').assign(family=lambda x:le.fit_transform(x.family))
y1 = y.stack('family')
display(y1)


# In[21]:


le = LabelEncoder()
X = (X
    .stack('family')  # wide to long
    .reset_index('family')  # convert index to column
    .assign(family=lambda x: le.fit_transform(x.family))  # label encode
)
y = y.stack('family')  # wide to long

display(y)


# # 4) Forecast with the DirRec strategy
# 
# Instatiate a model that applies the DirRec strategy to XGBoost.

# In[22]:


from sklearn.multioutput import RegressorChain

# YOUR CODE HERE
# model = ____
linreg = LinearRegression(fit_intercept=False)
xgb = XGBRegressor()
model = RegressorChain(base_estimator=xgb)

# Check your answer
q_4.check()


# In[23]:


# Lines below will give you a hint or solution code
#q_4.hint()
#q_4.solution()


# Run this cell if you'd like to train this model.

# In[24]:


model.fit(X, y)

y_pred = pd.DataFrame(
    model.predict(X),
    index=y.index,
    columns=y.columns,
).clip(0.0)


# And use this code to see a sample of the 16-step predictions this model makes on the training data.

# In[25]:


FAMILY = 'BEAUTY'
START = '2017-04-01'
EVERY = 16

y_pred_ = y_pred.xs(FAMILY, level='family', axis=0).loc[START:]
y_ = family_sales.loc[START:, 'sales'].loc[:, FAMILY]

fig, ax = plt.subplots(1, 1, figsize=(11, 4))
ax = y_.plot(**plot_params, ax=ax, alpha=0.5)
ax = plot_multistep(y_pred_, ax=ax, every=EVERY)
_ = ax.legend([FAMILY, FAMILY + ' Forecast'])


# # Next Steps #
# 
# Congratulations! You've completed Kaggle's *Time Series* course. If you haven't already, join our companion competition: [Store Sales - Time Series Forecasting](https://www.kaggle.com/c/29781) and apply the skills you've learned.
# 
# For inspiration, check out Kaggle's previous forecasting competitions. Studying winning competition solutions is a great way to upgrade your skills.
# 
# - [**Corporación Favorita**](https://www.kaggle.com/c/favorita-grocery-sales-forecasting): the competition *Store Sales* is derived from.
# - [**Rossmann Store Sales**](https://www.kaggle.com/c/rossmann-store-sales)
# - [**Wikipedia Web Traffic**](https://www.kaggle.com/c/web-traffic-time-series-forecasting/)
# - [**Walmart Store Sales**](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting)
# - [**Walmart Sales in Stormy Weather**](https://www.kaggle.com/c/walmart-recruiting-sales-in-stormy-weather)
# - [**M5 Forecasting - Accuracy**](https://www.kaggle.com/c/m5-forecasting-accuracy)
# 
# # References #
# 
# Here are some great resources you might like to consult for more on time series and forecasting. They all played a part in shaping this course:
# 
# - *Learnings from Kaggle's forecasting competitions*, an article by Casper Solheim Bojer and Jens Peder Meldgaard.
# - *Forecasting: Principles and Practice*, a book by Rob J Hyndmann and George Athanasopoulos.
# - *Practical Time Series Forecasting with R*, a book by Galit Shmueli and Kenneth C. Lichtendahl Jr.
# - *Time Series Analysis and Its Applications*, a book by Robert H. Shumway and David S. Stoffer.
# - *Machine learning strategies for time series forecasting*, an article by Gianluca Bontempi, Souhaib Ben Taieb, and Yann-Aël Le Borgne.
# - *On the use of cross-validation for time series predictor evaluation*, an article by Christoph Bergmeir and José M. Benítez.
# 
