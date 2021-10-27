# |------------------------------------------------------------------
# | # Sales Forecast  - Time Series Analysis TS3
# |------------------------------------------------------------------
# |
# | ## 1. Introduction
# |
# | This is a notebook to practice the routine procedures
# | commonly used in the time sequence analysis.
# | This notebook is based on the Kaggle [Time Series Analysis](https://www.kaggle.com/learn/time-series)
# | offered by Ryan Holbrook.

# | We have [the sales record of thousand types of products
# | in the supermarket-chain 'Favorita' in Ecuador](https://www.kaggle.com/c/store-sales-time-series-forecasting).
# | We will make predictions of the sales in the future, using
# | the data and machine learning techniques. We also have
# | anchillery data, such as the official holidays in Ecuador,
# | and the prices of oil in the country in the month.
# | The main focus in this notebook is __cycles__ in time series.
# | 'Lags' is the usual technique to deal with cycles, but we will
# | also try __moving average__ to reproduce the cycles, to see
# | how the two technique work similarly or differently.

# | There are two types of machine learning models.
# |  1. can learn trend  [linear regression].
# |  2. cannot learn trend [xgboost, decision trees, etc.].
# | because the latter do not learn to 'extrapolate'.
# |
# | In order to make the most of machine learning technique,
# | we will combine both. Use a linear model to extrapolate in the first stage,
# | and use more soffisticated models to reproeuce the residuals in the second
# | stage. This multi-stage stragey is called __hybrid models__.
# | In this notebook we will combine a linear regressor with `xgboost`.
# |

# | ## 2. Task
# | 1. Concentrate on the sales of magazines.
# |    + deseason.
# |    + lags.
# |    + partial Autocorrelation function.
# |    + anchillery data for the products on promotion.
# |
# | Note that lags contains seasonality, as 7-days (weekly) trend
# | shows up exactly as a high correlation of data separated by 7-days
# | (=7 days lag). In order to isolate the lagged features (that are not seanality),
# |  we will have to first remove the seasonality from the target feature.
# |
# | ## 3. Data
# | 1. [sales record in supermarket-chain 'Favorita' in Ecuador](https://www.kaggle.com/c/store-sales-time-series-forecasting).
# |


# -------------------------------------------------------
# |- cycles: lags
# |+ lags includes saesonality
# |+ remove them first
# |
# |- pacf
# |
# |- magazine with on promotion
# |- running average
# |
# |the number of items on a special promotion that day
# |
# |- rolling statistics
# |center = False
# -------------------------------------------------------

# | ## 4. Notebook
# -------------------------------------------------------
# | Import packages.

from pathlib import Path
import os
import pandas as pd
import numpy as np
from pandas import tseries
from datetime import datetime

from scipy.signal import periodogram
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor, RegressorChain

from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import pacf
import statsmodels.api as sm
from xgboost import XGBRegressor

from IPython.display import display
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import kaleido
from kaggle_tsa.ktsa import *

# Model 1
# from pyearth import Earth
# multivariate adaptive regression splines
# installation failed

from sklearn.linear_model import ElasticNet, Lasso, Ridge

# Model 2
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor


# -------------------------------------------------------
# | Set up directories.

CWD = Path('/Users/meg/git7/sales/')
DATA_DIR = Path('../input/store-sales-time-series-forecasting')
KAGGLE_DIR = Path('store-sales-time-series-forecasting')
IMAGE_DIR = Path('./images')
os.chdir(CWD)
set_cwd(CWD)

# -------------------------------------------------------
# | If the data is not downloaded yet, do so now.

set_data_dir(KAGGLE_DIR, CWD)
show_whole_dataframe(True)

# -------------------------------------------------------
# | Goal is to make 3 DafaFrames.
# |
# | 1. `sales` : whole table with minimum manipulation.
# | 2. `family`: sales aggregatedd for each product-family.
# | 3. `mag`   : `MAGAZINE` column of the data 2.

# -------------------------------------------------------
# | Read `sales` as it is.

sales = pd.read_csv(DATA_DIR/'train.csv')
print(sales.info())
s_cols = list(sales.columns)
s_cols.remove('id')

# -------------------------------------------------------
# | Read `sales` once again with manipulation.
# | Note
# | * The order of `MultiIndex` must be exactly like this.
# | * Need `.sort_index()`
# | Anomally on the new years day in 2017 is removed.

sales = pd.read_csv(DATA_DIR/'train.csv',
                    usecols=s_cols,
                    dtype={'store_nbr': 'category',
                           'family': 'category'},
                    parse_dates=['date'],
                    index_col='date')

sales = sales.loc[sales.index != pd.Timestamp(2017, 1, 1), :]
sales.index = sales.index.to_period('D')
sales.set_index(['store_nbr', 'family'], append=True, inplace=True)
sales.index = sales.index.reorder_levels(['store_nbr', 'family', 'date'])
sales = sales.sort_index()

# -------------------------------------------------------
# | Create `family` from `sales`.
# | Get used to
# | * `.unstack()` to pivot a `DataFrame`
# | * `.loc['2017']` to select rows without specifying
# |    minimun and maximum.
# |
# | The sales of all stores are avearged. Therefore
# | no column for `store_nbr` in `family`.

family = sales.groupby(['family', 'date']).mean().unstack(
    'family').loc['2017', ['sales', 'onpromotion']]

# -------------------------------------------------------
# | Create `mag` from `family`.
# | Note how to retrieve columns when the column indices have
# | multiple layers.

print(family.info())
mag = family.loc[:,
                 [('sales', 'MAGAZINES'), ('onpromotion', 'MAGAZINES')]]

# -------------------------------------------------------
# | Let us have a look at data.

trace = go.Scatter(x=mag.index.to_timestamp(),
                   y=mag['sales']['MAGAZINES'])
data = [trace]
layout = go.Layout(height=512,
                   font=dict(size=16),
                   showlegend=False)
fig = go.Figure(data=data, layout=layout)
fig_wrap(fig, IMAGE_DIR/'fig1.png')

# -------------------------------------------------------
# Looks like `mag` has a clean, week-long seasonality.
#
# -------------------------------------------------------
# As usual, we will check
# - periodogram (Fourier decomposition)
# - correlogram (PACF)

# -------------------------------------------------------
# | First perodogram.
# |
# |__`periodogram`__:\
# | `detrend` {'linear', 'constant'}\
# | `window`  {'boxcar', 'gaussian', ...} `scipy.signal.get_window`[shape of window function](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html#scipy.signal.get_window)\
# | `scaling` {'density', 'spectrum'}\
# | The units are [V<sup>2</sup>/Hz] for 'density' (power spectrum) and [V<sup>2</sup>] for 'spectrum'.

y = mag[('sales', 'MAGAZINES')].copy()
fig = create_periodogram(y)
fig_wrap(fig, IMAGE_DIR/'fig2.png')

# -------------------------------------------------------
# | Probably 4 components in a month would suffice.

fourier = CalendarFourier(freq='M', order=4)
dp = DeterministicProcess(
    constant=True,
    index=y.index,
    order=1,
    seasonal=True,  # period is taken from index frequency
    drop=True,
    additional_terms=[fourier],)

X = dp.in_sample()

y, X = y.align(X, join='inner', axis=0)

# -------------------------------------------------------
# | Train a linear regression model.

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, shuffle=False)

model = LinearRegression(fit_intercept=False)
model.fit(X_train, y_train)
y_fit = model.predict(X_train)
y_pred = model.predict(X_test)

# -------------------------------------------------------
# | Check the training results.

fig = show_training_results(X, y, X_train, y_fit, X_test, y_pred,
                            titles=('[Month]', '[Sales]',
                                    'Magazine sales'))

fig_wrap(fig, IMAGE_DIR/'fig3.png')
train_rmse, test_rmse = evaluate_error(
    y_train, y_fit, y_test, y_pred)

# -------------------------------------------------------
# | Now we will use `xgboost` as the second stage.
# -------------------------------------------------------
#
y_fit_ser = pd.Series(y_fit,
                      index=X_train.index.to_period('D'),
                      name='MAGAZINES')

y_res = y_train - y_fit_ser

xgb = XGBRegressor(max_depth=2,
                   gamma=1,
                   min_child_weight=15,

                   colsample_bytree=0.1,
                   subsample=0.1,

                   reg_alpha=0.0,
                   learning_rate=0.3)

xgb.fit(X_train, y_res)

y_fit_boosted = xgb.predict(X_train) + y_fit
y_pred_boosted = xgb.predict(X_test) + y_pred

fig = show_training_results(X, y,
                            X_train, y_fit_boosted,
                            X_test, y_pred_boosted,
                            titles=('[Month]', '[Sales]',
                                    'Magazine sales (Hybrid XGBoost)'))

fig_wrap(fig, IMAGE_DIR/'fig4.png')
train_rmse, test_rmse = evaluate_error(
    y_train, y_fit_boosted, y_test, y_pred_boosted)

# -------------------------------------------------------
# | After manually tunign the `xgboost` parameters,
# | we could go a good as a model with a linear regression only.


# | So, there is no advantage in adding `XGBoost` in the second
# | stage. We will try further folowing.
# |
# | 1. Create 'BoostedHybrid` class (in `ktsa.py` module)
# |    so that we can try different combinations of a linear model
# |    and a booster.
# |
# | 2. Create MA (moving average) features and feed them
# |    to the hybrid models.
# |

# -------------------------------------------------------
# | # Model 1 (Linear Model)
# | ```
# | from sklearn.linear_model import ElasticNet, Lasso, Ridge, LinearRegression
# | ```

# | # Model 2 (Booster)
# | ```
# |
# | from xgboost import XGBRegressor
# | from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
# | from sklearn.neighbors import KNeighborsRegressor
# | from sklearn.neural_network import MLPRegressor
# |
# | ```
# -------------------------------------------------------
# | Create features for the first stage.
# |

y = pd.DataFrame(mag[('sales', 'MAGAZINES')].copy())

fourier = CalendarFourier(freq='M', order=4)
dp = DeterministicProcess(
    constant=True,
    index=y.index,
    order=1,
    seasonal=True,  # period is taken from index frequency
    drop=True,
    additional_terms=[fourier],)

X1 = dp.in_sample()

X1_train, X1_test, y_train, y_test = train_test_split(
    X1, y, test_size=0.2, shuffle=False)

# -------------------------------------------------------
# | Create features for the second stage.

X2 = pd.DataFrame(mag[('onpromotion', 'MAGAZINES')].copy())
X2['day'] = X2.index.day

X2_train, X2_test, y_train, y_test = train_test_split(
    X2, y, test_size=0.2, shuffle=False)


# | Need some experiments to see if we should include
# | Fourier decomposition and deseasoning in the first stage
# | or second stage.
# -------------------------------------------------------

# -------------------------------------------------------
# | Try all combintations we can think of.
# | Do not forget to __instantiate__ the models!
# | [=add '()' after the function name]
# |

model_1_list = [LinearRegression(fit_intercept=False),
                ElasticNet(), Lasso(), Ridge()]

model_2_list = [XGBRegressor(), ExtraTreesRegressor(), RandomForestRegressor(),
                KNeighborsRegressor(), MLPRegressor()]

model_comb = [(m1, m2) for m1 in model_1_list for m2 in model_2_list]


train_rmse_list = []
test_rmse_list = []
comb_list = []

for m1, m2 in model_comb:

    m1_name = str(m1.__class__()).split('(')[0]
    m2_name = str(m2.__class__()).split('(')[0]
    print(f'{m1_name} {m2_name}')

    model = BoostedHybrid(m1, m2)
    model.fit(X1_train, X2_train, y_train)

    y_fit = model.predict(X1_train, X2_train)
    y_pred = model.predict(X1_test, X2_test)

    train_rmse, test_rmse = evaluate_error(
        y_train, y_fit, y_test, y_pred)

    train_rmse_list.append(train_rmse)
    test_rmse_list.append(test_rmse)
    comb_list.append((m1_name, m2_name))

    print('#'+'='*20)

hybrid_result = pd.DataFrame(
    dict(train_rmse=train_rmse_list,
         test_rmse=test_rmse_list,
         combination=comb_list)
)

display(hybrid_result.sort_values('test_rmse'))

# | The best combination so far is
# | * __`Lasso`__ and __`RandomForestRegressor`__
# |
# -------------------------------------------------------
# | Let us have a look.
# |

model = BoostedHybrid(Lasso(), RandomForestRegressor())
model.fit(X1_train, X2_train, y_train)

y_fit = model.predict(X1_train, X2_train)
y_pred = model.predict(X1_test, X2_test)

fig = show_training_results(X, y,
                            X1_train, y_fit.values,
                            X1_test, y_pred.values,
                            titles=('[Month]', '[Sales]',
                                    'Magazine sales (Optimal Hybrid)'))

fig_wrap(fig, IMAGE_DIR/'fig5.png')

# -------------------------------------------------------
# | Looks okay.


# -------------------------------------------------------
# | Here we will add moving average and other
# | statistical features. First check how
# | moving average compares with the target.
# | Quickly repeat modeling with linear regression.
# |
# |

y = mag[('sales', 'MAGAZINES')].copy()

fourier = CalendarFourier(freq='M', order=4)
dp = DeterministicProcess(
    constant=True,
    index=y.index,
    order=1,
    seasonal=True,  # period is taken from index frequency
    drop=True,
    additional_terms=[fourier],)

X = dp.in_sample()

# y, X = y.align(X, join='inner', axis=0)

# -------------------------------------------------------
# Split the data

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, shuffle=False)

# -------------------------------------------------------
# Train the model.

model = LinearRegression(fit_intercept=False)
model.fit(X_train, y_train)

y_fit = model.predict(X_train)
y_pred = model.predict(X_test)

# -------------------------------------------------------
# Show the results.

y_fit_ser = pd.Series(y_fit,
                      index=X_train.index,
                      name='MAGAZINES')

y_res = y_train - y_fit_ser

y_ma = y_train.rolling(window=7, center=True).mean().fillna(0.0)

trace_1 = go.Scatter(x=X_train.index.to_timestamp(),
                     y=y_train, name='Training Data')

trace_2 = go.Scatter(x=X_train.index.to_timestamp(),
                     y=y_res, name='Residual')

trace_3 = go.Scatter(x=X_train.index.to_timestamp(),
                     y=y_ma, name='Moving Average')

data = [trace_1, trace_2, trace_3]

layout = go.Layout(height=640,
                   font=dict(size=16))

fig = go.Figure(data=data, layout=layout)
fig_wrap(fig, IMAGE_DIR/'fig6.png')

# -------------------------------------------------------
# | It is indeed the that the moving average reproduce
# | the residual good. It would be helpful to inclue the
# | moving average in the features.
# | We will create new features with moving aggregation
# | together with the ones we know already, time dummy
# | and lags.
# |

# -------------------------------------------------------
# | Check how many lags are effective
# |
y = mag[('sales', 'MAGAZINES')].copy()
n_lag = 12  # safe large value..
n_cols = 3  # for plotting.

fig, corr = create_lag_plot(y, n_lag, n_cols)
fig_wrap(fig, IMAGE_DIR/'fig7.png')
print(corr)

# -------------------------------------------------------
# | PACF.

fig = create_pacf_plot(y, n_lag)
fig_wrap(fig, IMAGE_DIR/'fig8.png')

# -------------------------------------------------------
# | We will  take then `n_lag=10`.
# -------------------------------------------------------
# | Let us quickly write function to create
# | lags and steps.

n_lag = 10
X_lag = make_lagged_features(y, n_lag)

# We do not use it here.
# n_step = 8
# y = make_step_features(y, n_step)

# -------------------------------------------------------
# | Check also Fourier decomposition.

fig = create_periodogram(y)
fig_wrap(fig, IMAGE_DIR/'fig9.png')

# -------------------------------------------------------
# | Okay, `order = 4`.
# -------------------------------------------------------

fourier = CalendarFourier(freq='M', order=4)
dp = DeterministicProcess(
    constant=True,
    index=y.index,
    order=1,
    seasonal=True,  # period is taken from index frequency
    drop=True,
    additional_terms=[fourier],)

X_time = dp.in_sample()

# -------------------------------------------------------
# | `onpromotion` fetures
# | __IMPORTANT:__
# |
# | The company can decide when to have a promotion
# | as they like. The decisions are made usually well
# | beforehand, therefore one can use 'future' `onpromotion`
# | information in the prediction.
# |

y_promo = mag[('onpromotion', 'MAGAZINES')].copy()
n_lag = 2
X_promo_lag = make_lagged_features(y_promo, n_lag)

n_step = 2
X_promo_step = make_step_features(y_promo, n_step)

# -------------------------------------------------------
# | Moving average features.
# |
# | __IMPORTANT:__
# | Do not 'center' the rolling average. Otherwise
# | we are using future sales information that should not be
# | available at the time of the forecast (= leakage).
# | Just set  `center = False` (default). As we discussed
# | before, this does not apply to the promotion information.
# |

X_mean_7 = y.shift(1).rolling(7, center=False).mean().fillna(0.0)
X_median_14 = y.shift(1).rolling(14, center=False).median().fillna(0.0)
X_std_7 = y.shift(1).rolling(7, center=False).std().fillna(0.0)
X_promo_7 = y.shift(1).rolling(7, center=True).sum().fillna(0.0)

# -------------------------------------------------------
# | Moving average features.
# | Now we have following featuers.
# |
# | - `X_time`
# | - `X_lag`
# |
# | - `X_promo_lag`
# | - `X_promo_step'
# |
# | - 'X_mean_7'
# | - 'X_median_14'
# | - 'X_std_7'
# | - 'X_promo_7'
# |

# |  Let us give 'X_time' for the first stage models,
# |  and all the rest ot the second stage.
# |
# |

X1 = X_time.copy()
X2 = pd.concat([X_lag,
                X_promo_lag,
                X_promo_step,
                X_mean_7,
                X_median_14,
                X_std_7,
                X_promo_7], axis=1)

X2['day'] = X2.index.day

# -------------------------------------------------------
# `y` must be `pd.DataFrame` not `pd.Series`.

y = pd.DataFrame(y)

# -------------------------------------------------------
# `feture_names must be unique`.
#

print(X1.info())
print(X2.info())

X2.columns = ['col_'+str(i) for i in range(len(X2.columns))]
X2.head(3)


X1_train, X1_test, y_train, y_test = train_test_split(
    X1, y, test_size=0.2, shuffle=False)

X2_train, X2_test, y_train, y_test = train_test_split(
    X2, y, test_size=0.2, shuffle=False)

# -------------------------------------------------------
# Hybrid solutions.

model_1_list = [LinearRegression(fit_intercept=False),
                ElasticNet(), Lasso(), Ridge()]

model_2_list = [XGBRegressor(), ExtraTreesRegressor(), RandomForestRegressor(),
                KNeighborsRegressor(), MLPRegressor()]

model_comb = [(m1, m2) for m1 in model_1_list for m2 in model_2_list]

train_rmse_list = []
test_rmse_list = []
comb_list = []

for m1, m2 in model_comb:

    m1_name = str(m1.__class__()).split('(')[0]
    m2_name = str(m2.__class__()).split('(')[0]
    print(f'{m1_name} {m2_name}')

    model = BoostedHybrid(m1, m2)
    model.fit(X1_train, X2_train, y_train)

    y_fit = model.predict(X1_train, X2_train)
    y_pred = model.predict(X1_test, X2_test)

    train_rmse, test_rmse = evaluate_error(
        y_train, y_fit, y_test, y_pred)

    train_rmse_list.append(train_rmse)
    test_rmse_list.append(test_rmse)
    comb_list.append((m1_name, m2_name))

    print('#'+'='*20)

hybrid_result = pd.DataFrame(
    dict(train_rmse=train_rmse_list,
         test_rmse=test_rmse_list,
         combination=comb_list)
)

display(hybrid_result.sort_values('test_rmse'))

# -------------------------------------------------------
# | Again the best is the combination of
# | * __`Lasso`__ and __`RandomForestRegressor`__.
# |
# | While there many combinations that apparently suffer
# | from overfitting, `Lasso`+`RandomForestRegressor`
# | has a good balance in training and test errors.
# |
# | Let us have a look.
# |

model = BoostedHybrid(Lasso(), RandomForestRegressor())
model.fit(X1_train, X2_train, y_train)

y_fit = model.predict(X1_train, X2_train)
y_pred = model.predict(X1_test, X2_test)

fig = show_training_results(X, y,
                            X1_train, y_fit.values,
                            X1_test, y_pred.values,
                            titles=('[Month]', '[Sales]',
                                    'Magazine sales (Optimal Hybrid)'))

fig_wrap(fig, IMAGE_DIR/'fig10.png')

# -------------------------------------------------------
# | Looks excellent.
# -------------------------------------------------------
# | END
