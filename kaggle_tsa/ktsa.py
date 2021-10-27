import pandas as pd
import numpy as np
# import geopandas as gpd

from pathlib import Path
import os
import webbrowser
import zipfile

import plotly.graph_objs as go
from plotly.subplots import make_subplots
# import folium
# from folium import Choropleth, Circle, Marker
# from folium.plugins import HeatMap, MarkerCluster, HeatMap
from IPython.display import IFrame

from scipy.signal import periodogram
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import pacf
import pdb

# -------------------------------------------------------
# | Some housekeeping functions.
# | Later they will go to a module ``../kaggle_geospatial`.


def set_cwd(CWD):
    if Path.cwd() != CWD:
        os.chdir(CWD)

# | If we have not downloaded the course data, get it from Alexis Cook's
# | kaggle public dataset.


def set_data_dir(KAGGLE_DIR,  CWD):
    '''

    KAGGLE_DIR : Path() -  for the data source on the internet
    CWD : Path() - current working directory.

    This function assumes to be executed from CWD

    '''
    os.chdir(CWD)
    INPUT_DIR = Path('../input')

    if not (INPUT_DIR/KAGGLE_DIR.stem).exists():
        (INPUT_DIR/KAGGLE_DIR.stem).mkdir(parents=True)
        os.chdir(INPUT_DIR)

        command = 'kaggle d download ' + str(KAGGLE_DIR)
        x = os.system(command)
        if x == 256:
            command = 'kaggle c download ' + str(KAGGLE_DIR)
            x = os.system(command)

        zip_list = list(Path('.').glob('*.zip'))

        for z in zip_list:
            z_path = Path(str(z).replace('.zip', ''))
#            z_path.mkdir(exist_ok=True)
            z = z.rename(z_path/z)
#            print(f'\033[31mz: {z}\033[0m')
            with zipfile.ZipFile(z, 'r') as zip_ref:
                zip_ref.extractall(z_path)

        os.chdir(str(CWD))

# -------------------------------------------------------
# | Some housekeeping stuff. Change `pandas`' options so that we can see
# | whole DataFrame without skipping the lines in the middle.


def show_whole_dataframe(show):
    if show:
        pd.options.display.max_rows = 999
        pd.options.display.max_columns = 99

# -------------------------------------------------------
# to show plotly plot on jupyter and browser as well.


def fig_wrap(fig, file_name):
    w = '1280px'
    h = '640px'
    from IPython import get_ipython
    fig.write_image(file_name, width=w, height=h)

    is_jupyter = get_ipython().__class__.__name__
    if is_jupyter == 'NoneType':
        return fig.show()

    else:
        from IPython.display import IFrame
        html_file = Path(str(file_name).replace('png', 'html'))
        fig.write_html(html_file)
        return IFrame(html_file, width='100%', height='640px')
#        return IFrame(file_name, width=w, height=h)
# -------------------------------------------------------


def embed_map(m, file_name):
    from IPython.display import IFrame
    m.save(file_name)
    return IFrame(file_name, width='100%', height='500px')


def embed_plot(fig, file_name):
    from IPython.display import IFrame
    fig.write_html(file_name)
    return IFrame(file_name, width='100%', height='500px')


def show_on_browser(m, file_name):
    '''
    m   : folium Map object
    Do not miss the trailing '/'
    '''
    m.save(file_name)
    url = 'file://'+file_name
    webbrowser.open(url)

# -------------------------------------------------------
# split Week


def split_week(ts, append=True):

    a = []
    _ = [a.append(pd.Timestamp(i.split('/')[0])) for i in ts['Week']]
    b = []
    _ = [b.append(pd.Timestamp(i.split('/')[1])) for i in ts['Week']]
    ts['Week_Begin'] = a
    ts['Week_End'] = b

    ts.set_index(['Week_Begin'], append=append, inplace=True)

    return ts


def is_index_continuous(ts, freq='W-SUN'):

    # in case of MultiIndex
    t_min = ts.index.get_level_values(0).min()
    t_max = ts.index.get_level_values(0).max()
    idx_dt = pd.date_range(start=t_min, end=t_max, freq=freq)

    try:
        return (ts.index.get_level_values(0) == idx_dt).mean() == 1.0
    except ValueError:
        return False

# -------------------------------------------------------
# | Create a periodogram.


def create_periodogram(y) -> go.Figure():

    annual_freq = (
        'Annual (1)',
        'Semiannual (2)',
        'Quarterly (4)',
        'Bimonthly (6)',
        'Monthly (12)',
        'Biweekly (26)',
        'Weekly (52)',
        'Semiweekly (104)')

    annual_interval = [1, 2, 4, 6, 12, 26, 52, 104]

    fs = pd.Timedelta('1Y')/pd.Timedelta('1D')
    frequencies, spectrum = periodogram(y,
                                        fs=fs,
                                        detrend='linear',
                                        window='boxcar',
                                        scaling='spectrum'
                                        )

    trace = go.Scatter(x=frequencies,
                       y=spectrum,
                       fill='tozeroy',
                       line=dict(color='coral'),
                       line_shape='hvh')

    data = [trace]
    layout = go.Layout(height=640,
                       font=dict(size=20),
                       xaxis=dict(type='log',
                                  ticktext=annual_freq,
                                  tickvals=annual_interval))

    fig = go.Figure(data=data, layout=layout)
    return fig


# -------------------------------------------------------
# | Create a lag plot.


def create_lag_plot(y, n_lag, n_cols) -> (go.Figure(), list):

    n_rows = n_lag // n_cols
    sm.OLS.df_degree = 1

    fig = make_subplots(cols=n_cols,
                        rows=n_rows,
                        vertical_spacing=0.04,
                        subplot_titles=[f'Lag {i}' for i in range(1, n_lag+1)]
                        )

    xax = ['x' + str(i) for i in range(1, n_lag+1)]
    xax[0] = 'x'

    yax = ['y' + str(i) for i in range(1, n_lag+1)]
    yax[0] = 'y'

    trace = [go.Scatter(x=y,
                        y=y.shift(i),
                        marker=dict(opacity=0.7),
                        mode='markers',
                        xaxis=xax[i-1],
                        yaxis=yax[i-1]) for i in range(1, n_lag+1)]

    trace_reg = [go.Scatter(x=sorted(y),
                            y=sorted(sm.OLS(y.shift(i).values, y,
                                            missing='drop').fit().fittedvalues),
                            mode='lines',
                            xaxis=xax[i-1], yaxis=yax[i-1]) for i in range(1, n_lag+1)]

    data = trace + trace_reg
    layout = go.Layout(height=1024,
                       font=dict(size=20),
                       showlegend=False)

    layout = fig.layout.update(layout)
    fig = go.Figure(data=data, layout=layout)
    corr = [y.autocorr(lag=i) for i in range(1, n_lag+1)]

    return fig, corr

# -------------------------------------------------------
# | Create a PACF plot. (=Partial Autocorrelation Function)


def create_pacf_plot(y, n_lag) -> go.Figure():

    x_pacf, x_conf = pacf(y, nlags=n_lag, alpha=0.05)
    x_error = (x_conf - x_pacf.repeat(2).reshape(x_conf.shape))[:, 1]
    x_sig = 1.96 / np.sqrt(len(y))

    lags_name = [f'Lag {i-1}' for i in range(1, n_lag+1)]

    trace_1 = go.Bar(y=lags_name,
                     error_x=dict(type='data', array=x_error),
                     x=x_pacf,
                     marker_color='indianred',
                     opacity=0.8,
                     width=0.8,
                     orientation='h')

    layout = go.Layout(height=512, width=800,
                       font=dict(size=20),
                       showlegend=False,
                       xaxis=dict(range=[-1.1, 1.1]))

    data = [trace_1]

    fig = go.Figure(data=data, layout=layout)
    fig.add_vrect(x0=-x_sig, x1=x_sig,
                  fillcolor='teal',
                  line_color='teal',
                  opacity=0.3,
                  line_width=2)

    return fig


# -------------------------------------------------------
# Quickly show training results


def show_training_results(X, y, X_train, y_fit, X_test, y_pred,
                          titles=('[Year]', '[Cases]',
                                  'Flu-visit predictions in 2015-2016')):
    '''
    y : pd.DataFrame or pd.Series
    y_fit: np.array
    y_pred: np.array

    '''

    try:
        X.index = X.index.to_timestamp()
    except (AttributeError, TypeError):
        pass

    try:
        X_train.index = X_train.index.to_timestamp()
    except (AttributeError, TypeError):
        pass

    try:
        X_test.index = X_test.index.to_timestamp()
    except (AttributeError, TypeError):
        pass

    try:  # in case of multiple output

        n_step = y.shape[1]

        trace_1 = go.Scatter(x=X.index,
                             y=y.values[:, 0], name='Truth', mode='lines+markers')
        data = [trace_1]
        for i in range(n_step):

            trace_2 = go.Scatter(x=X_train.index,
                                 y=y_fit[:, i], name='Training')

            trace_3 = go.Scatter(x=X_test.index,
                                 y=y_pred[:, i], name='Forecast')

            data.extend([trace_2, trace_3])
            y_min = y.values[:, 0].min()
            y_max = y.values[:, 0].max()

    except IndexError:  # y is not 2D.

        trace_1 = go.Scatter(x=X.index,
                             y=y, name='Truth', mode='lines+markers')

        trace_2 = go.Scatter(x=X_train.index,
                             y=y_fit, name='Training')

        trace_3 = go.Scatter(x=X_test.index,
                             y=y_pred, name='Forecast')

        data = [trace_1, trace_2, trace_3]
        y_min = y.values.min()
        y_max = y.values.max()

    layout = go.Layout(height=640,
                       font=dict(size=16),
                       xaxis=dict(title=titles[0]),
                       yaxis=dict(title=titles[1],
                                  autorange=False,
                                  range=[y_min, y_max]),
                       title_text=titles[2])

    fig = go.Figure(data=data, layout=layout)

    return fig
# -------------------------------------------------------
# evaluate root mean squared error


def evaluate_error(y_train, y_fit, y_test, y_pred):

    train_rmse = mean_squared_error(y_train, y_fit, squared=False)
    print(f'\033[33mTrain RMSE: \033[96m{train_rmse:6.2f}\033[0m')

    test_rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f'\033[91mTest RMSE : \033[96m{test_rmse:6.2f}\033[0m')

    return train_rmse, test_rmse


# -------------------------------------------------------
# class test
class MyClass:
    def __init__(self, a, b) -> None:
        self.a = a
        self.b = b

    def fit(self, X1, y) -> float:
        print(self.a.__name__)
        print(self.b.__name__)
        y_fit = (self.a).fit(X1, y)

        return len(X1), len(y)

# -------------------------------------------------------
# hybrid model class


class BoostedHybrid:

    def __init__(self, model_1, model_2) -> None:

        self.model_1 = model_1
        self.model_2 = model_2

        self.y_columns = None

    def fit(self, X_1, X_2, y) -> None:

        self.model_1.fit(X_1, y)

        y_fit = pd.DataFrame(
            self.model_1.predict(X_1),
            index=X_1.index, columns=y.columns,
        )

        y_resid = y - y_fit
        y_resid = y_resid.stack().squeeze()  # wide to long

        self.model_2.fit(X_2, y_resid)

        # Save column names for predict method
        self.y_columns = y.columns

        # Save data for question checking
        self.y_fit = y_fit
        self.y_resid = y_resid

    def predict(self, X_1, X_2) -> pd.DataFrame:
        y_pred = pd.DataFrame(

            self.model_1.predict(X_1),
            index=X_1.index, columns=self.y_columns,
        )
        y_pred = y_pred.stack().squeeze()  # wide to long

        y_pred += self.model_2.predict(X_2)

        return y_pred.unstack()  # long to wide

# -------------------------------------------------------
# lag features
# -------------------------------------------------------


def make_lagged_features(y, n_lag):

    return pd.concat({f'y_lag_{i}': y.shift(i) for i in range(1, n_lag+1)},
                     axis=1).fillna(0.0)

# -------------------------------------------------------
# step features (for multipe output)
# -------------------------------------------------------


def make_step_features(y, n_step):

    return pd.concat({f'y_step_{i}': y.shift(-i) for i in range(0, n_step)},
                     axis=1).fillna(0.0)

# -------------------------------------------------------
# END
# =======================================================
