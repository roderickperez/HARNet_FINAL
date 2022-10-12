from copy import copy
from util import HARNetCfg, get_MAN_data, year_range_to_idx_range
from model import scaler_from_cfg, model_from_cfg, get_loss, LRTensorBoard, MetricCallback, get_model_metrics, get_pred
from tensorflow.python.keras import backend as K
from pydantic.json import pydantic_encoder
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
import json
import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from plotly import graph_objects as go
from datetime import date, datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


pd.options.display.float_format = '{:,e}'.format
pd.options.display.width = 0

############################

plt.rcParams["figure.figsize"] = (6, 2)
plt.rcParams.update({'font.size': 6})
##############################
st.set_page_config(page_title="HARNet | UniWien Research Project",
                   page_icon=":chart_with_upwards_trend:",
                   layout="wide")

#########################
# ---- Header ----
with st.container():
    st.title(':chart_with_upwards_trend: HarNet App')

# Sidebar

st.sidebar.image("images/uniWienLogo.png", use_column_width=True)

# Create Config File
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    ['Preloaded Dataset', 'Data to Model', 'Model Parameters Summary', 'Metrics per Epoch', 'Metrics Plot', 'Metrics History', 'Forecast Plot', 'Forecast Data'])

#################################
dataSetExpander = st.sidebar.expander("Dataset", expanded=True)

dataSetOptions = dataSetExpander.selectbox(
    "Select Dataset", ['MAN', 'USEUINDXD', 'VIXCLS'])

filePath = "./data/MAN_data.csv"
# filePath = "./data/USEPUINDXD.csv"
# filePath = "./data/VIXCLS.csv"

stockOptions = dataSetExpander.selectbox(
    "Select Stock", ['.AEX', '.AORD', '.BFX', '.BSESN', '.BVSP', '.DJI', '.FCHI', '.FTSE',
                     '.GDAXI', '.HSI', '.IBEX', '.IXIC', '.KS11', '.MXX', '.NSEI', '.RUT',
                     '.SPX', '.SSEC', '.SSMI', '.STI', '.STOXX50E', '.KSE', '.N225', '.OSEAX',
                     '.GSPTSE', '.SMSI', '.OMXC20', '.OMXHPI', '.OMXSPI', '.FTMIB', '.BVLG'], 5)

variableSelection = dataSetExpander.selectbox("Select Variable", ['rsv', 'close_price', 'close_time',
                                                                  'rk_twoscale', 'bv_ss', 'medrv',
                                                                  'rsv_ss', 'rv10_ss', 'rv5_ss',
                                                                  'rk_th2', 'rk_parzen', 'bv',
                                                                  'rv10', 'open_price', 'open_time',
                                                                  'open_to_close', 'rv5', 'nobs'], 8)
# if st.sidebar.button("Preload Dataset"):
# def custom_date_parser(x): return datetime.strptime(x, "%Y-%m-%d %H:%M:%S%z%z")


df = pd.read_csv(filePath, sep=',')
df['Date'] = df['Date'].astype(str)
df[['Date', 'Time']] = df['Date'].str.split(" ", 1, expand=True)
#df.index = df['Date']
df = df.drop(['Time'], axis=1)

df_symbol = df.groupby(['Symbol'])
df = df_symbol.get_group(stockOptions)
df = df.drop('Symbol', axis=1)
df = df.reset_index(drop=True)

minYear = pd.to_datetime(df['Date'].min()).date()
maxYear = pd.to_datetime(df['Date'].max()).date()

timeExpander = st.sidebar.expander("Date Selection")

startPeriod = timeExpander.date_input("Start Date", minYear)
endPeriod = timeExpander.date_input("End Date", maxYear)

dateMask = (df['Date'] >= str(startPeriod)) & (df['Date'] <= str(endPeriod))
df_ = df.loc[dateMask]

trainingPercentage = timeExpander.slider('Training Percentage',
                                         min_value=0, max_value=100, value=80, step=5)
testPercentage = timeExpander.slider('Test Percentage',
                                     min_value=0, max_value=100, value=(100-trainingPercentage), step=5, disabled=True)

dateMask = (df['Date'] >= str(startPeriod)) & (df['Date'] <= str(endPeriod))

# df_ is a intermediate dataFrame created to then split it into train and test

df_ = df.loc[dateMask]
df_ = df_.reset_index(drop=True)

minIndex = df_.index.min()
maxIndex = df_.index.max()

totalIndex = maxIndex-minIndex
trainingPercentageIndex = round(totalIndex*(trainingPercentage/100))
testingPercentageIndex = totalIndex-trainingPercentageIndex

df_train = df_[:-testingPercentageIndex]
df_test = df_[-testingPercentageIndex:]

trainStartPeriod = df_train['Date'].values[0]
trainEndPeriod = df_train['Date'].values[-1]

testStartPeriod = df_test['Date'].values[0]
testEndPeriod = df_test['Date'].values[-1]

with tab1:
    st.subheader(f'Dataset: {dataSetOptions} | {stockOptions}')
    tab11, tab12, tab13 = st.tabs(['Data', 'Statistics', 'Plot'])

    with tab11:
        st.dataframe(df)

    with tab12:
        st.write('Initial Index: ' + str(minIndex))
        st.write('Last Index: ' + str(maxIndex))

        st.write('Total Samples: ', str(totalIndex))
        st.write('Training Samples: ', str(trainingPercentageIndex))
        st.write('Testing Samples: ', str(testingPercentageIndex))

    with tab13:
        #######################
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=[trainStartPeriod, trainStartPeriod, trainEndPeriod, trainEndPeriod, trainStartPeriod],
                       y=[df[variableSelection].min(), df[variableSelection].max(),
                          df[variableSelection].max(
                       ), df[variableSelection].min(),
                df[variableSelection].min()], fill="toself", opacity=0.3,
                mode="none", name=f"Train", fillcolor='green'))
        fig.add_trace(
            go.Scatter(x=[testStartPeriod, testStartPeriod, testEndPeriod, testEndPeriod, testStartPeriod],
                       y=[df[variableSelection].min(), df[variableSelection].max(),
                          df[variableSelection].max(
                       ), df[variableSelection].min(),
                df[variableSelection].min()], fill="toself", opacity=0.3,
                mode="none", name=f"Test", fillcolor='red'))
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df[variableSelection], name='Data', mode='lines', line=dict(color='blue')))
        fig.layout.update(
            xaxis_rangeslider_visible=True)
        fig.update_layout(
            autosize=False,
            width=1400,
            height=400,
            plot_bgcolor="black",
            margin=dict(
                l=50,
                r=50,
                b=0,
                t=0,
                pad=2
            ))
        st.plotly_chart(fig)

start_year_train = timeExpander.slider(
    'Select Start Year for Training:', min_value=2000, max_value=2022, value=2012, step=1)
n_years_train = timeExpander.slider(
    'Number of Year for Training:', min_value=1, max_value=21, value=4, step=1)
start_year_test = timeExpander.slider(
    'Select Start Year for Test:', min_value=2016, max_value=2022, value=2016, step=1, disabled=False)
n_years_test = timeExpander.slider(
    'Number of Years for Test:', min_value=1, max_value=6, value=1, step=1)


# dataSetExpanderdataSetExpander
modelParametersExpander = st.sidebar.expander("Model Parameters")

modelType = modelParametersExpander.selectbox(
    "Model Type:", ['HAR', 'HARSVJ', 'HARSVJR', 'HARNetSVJ', 'HARNet', 'HARNetSVJ', 'NaiveAvg'], 4)

filtersDConv = modelParametersExpander.number_input(
    'Filters De-Conv', min_value=1, value=1)
useBiasDeConv = modelParametersExpander.radio(
    'Use Bias De-Conv', [True, False], True)

activationFunction = modelParametersExpander.selectbox(
    'Activation Function', ['relu', 'tanh', 'sigmoid'], 0)
learningRate = modelParametersExpander.slider('Learning Rate',
                                              min_value=0.0001, max_value=1.0, value=0.0001, step=0.1, format='%.4f')
epochs = modelParametersExpander.slider(
    'Epochs', min_value=1, max_value=1000, value=100, step=1)
stepsPerEpoch = modelParametersExpander.slider(
    'Steps per Epoch', min_value=1, max_value=100, value=1, step=1)
labelLlength = modelParametersExpander.slider(
    'Label Length', min_value=1, max_value=100, value=5, step=1)
batchSize = modelParametersExpander.slider(
    'Batch Size', min_value=1, max_value=100, value=4, step=1)
optimizer = modelParametersExpander.selectbox(
    'Optimizer', ['Adam', 'RMSProp', 'SGD', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam'], 0)
loss = modelParametersExpander.selectbox('Loss', ['QLIKE'])
verbose = modelParametersExpander.selectbox('Verbose', [0, 1], 0)
baselineFit = modelParametersExpander.selectbox(
    'Baseline Fit', ['OLS', 'WLS'])

scalerParametersExpander = st.sidebar.expander("Scaler Parameters")

scalerSelection = scalerParametersExpander.selectbox(
    'Scaler', ['MinMax', 'LogMinMax', 'None'], 0)
scalerMin = scalerParametersExpander.slider('Scaler Minimum',
                                            min_value=0.0, max_value=1.0, value=0.0, step=0.1, format='%.3f')
scalerMax = scalerParametersExpander.slider('Scaler Maximum',
                                            min_value=0.0, max_value=1.0, value=0.001, step=0.1, format='%.3f')

otherParametersExpander = st.sidebar.expander("Other Parameters")
include_sv = otherParametersExpander.radio(
    'Include SV', [True, False], True)
save_best_weights = otherParametersExpander.selectbox(
    'Save Best Weights', [True, False], 1)
runEagerly = otherParametersExpander.radio(
    'Run Eagerly', [True, False], 1)
#####################

if st.sidebar.button('Execute Model'):
    # Get the configparser object
    configFile = {
        "model": modelType,
        "filters_dconv": filtersDConv,
        "use_bias_dconv": useBiasDeConv,
        "activation_dconv": activationFunction,
        "lags": [
            1,
            5,
            20
        ],
        "learning_rate": learningRate,
        "epochs": epochs,
        "steps_per_epoch": stepsPerEpoch,
        "label_length": labelLlength,
        "batch_size": batchSize,
        "optimizer": optimizer,
        "loss": loss,
        "verbose": verbose,
        "baseline_fit": baselineFit,
        "path": filePath,
        "asset": stockOptions,
        "include_sv": include_sv,
        "start_year_train": start_year_train,
        "n_years_train": n_years_train,
        "start_year_test": start_year_test,
        "n_years_test": n_years_test,
        "scaler": scalerSelection,
        "scaler_min": scalerMin,
        "scaler_max": scalerMax,
        "tb_path": "./tb/",
        "save_path": "./results/",
        "save_best_weights": save_best_weights,
        "run_eagerly": runEagerly
    }

    configFileJSON = json.dumps(configFile, skipkeys=False,
                                indent=4,
                                ensure_ascii=True,
                                check_circular=True,
                                allow_nan=True,
                                cls=None,
                                separators=None,
                                default=None,
                                sort_keys=False)
    # Write JSON file
    with open("config.in", "w") as jsonfile:
        jsonfile.write(configFileJSON)

    # Opening JSON file
    f = open('config.in')
    data = json.load(f)

    #######################################
    # parse command line
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", default="config.in", nargs='?',
                        help="Base configuration file to use (e.g. config.in). Must be a JSON-file of a config-dict. The used configuration is a combination of defaults set in util.py and parameters set in this file.")
    args = parser.parse_args()

    # load configuration
    cfg = HARNetCfg()
    if os.path.exists(args.cfg):
        with open(args.cfg) as f:
            cfg_in = json.load(f)
        for key in cfg_in:
            setattr(cfg, key, cfg_in[key])

    exp_name = os.path.splitext(os.path.basename(args.cfg))[0]
    logger = logging.getLogger('harnet')

    logger.info(f"Initializing experiment {exp_name}: {cfg.epochs} ...")
    logger.debug(json.dumps(cfg, indent=4, default=pydantic_encoder))

    save_path_curr = os.path.join(cfg.save_path, exp_name)
    tb_path_curr = os.path.join(cfg.tb_path, exp_name)

    if not os.path.exists(Path(cfg.save_path)):
        os.makedirs(Path(cfg.save_path))

    if not os.path.exists(Path(save_path_curr)):
        os.makedirs(Path(save_path_curr))

    # save full config
    with open(os.path.join(save_path_curr, "cfg_full.in"), 'w') as f:
        json.dump(cfg, f, indent=4, default=pydantic_encoder)

    # copy original config file
    shutil.copyfile(args.cfg, os.path.join(
        save_path_curr, os.path.basename(args.cfg)))

    # load data
    ts = get_MAN_data(cfg.path, cfg.asset, cfg.include_sv)

    # col1, col2 = st.columns(2)
    # with col1:
    #     st.write('Xandro')
    #     st.write(len(ts))
    #     st.write(ts)
    df = df.set_index('Date')
    df = df[variableSelection]
    # with col2:
    #     st.write('RPA')

    #     st.write(len(df))
    #     st.write(df)

    ts_ = copy(ts)
    ###################################
    if cfg.include_sv and "log" in cfg.scaler.lower():
        ts.iloc[:, -1] = (1 + ts.values[:, -1] / ts.values[:, 0])

    # normalize input time series
    scaler = scaler_from_cfg(cfg)

    ts_norm = pd.DataFrame(data=scaler.fit_transform(
        ts_.to_numpy()), index=ts.index)

    df_norm = pd.DataFrame(data=scaler.fit_transform(
        df.to_numpy()), index=df.index)

    ts_['norm'] = ts_norm.iloc[:, 0]
    ts_.index = pd.to_datetime(ts_.index).date

    # df['norm'] = df_norm
    #df.index = pd.to_datetime(df.index).date
    df = pd.concat([df, df_norm], axis=1, ignore_index=True)
    df = df.rename(columns={0: variableSelection, 1: 'norm'})

    # create train datasets
    year_range_train = [cfg.start_year_train,
                        cfg.start_year_train + cfg.n_years_train]
    year_range_test = [cfg.start_year_train + cfg.n_years_train,
                       cfg.start_year_train + cfg.n_years_train + cfg.n_years_test]

    # st.write(year_range_train)
    # st.write(year_range_test)

    # idx_range_train = year_range_to_idx_range(ts_norm, year_range_train)
    # idx_range_test = year_range_to_idx_range(ts_norm, year_range_test)

    # col1, col2 = st.columns(2)
    # with col1:
    #     st.write('Xandro')
    #     st.write(len(ts_))
    #     st.write(ts_)

    # with col2:
    #     st.write('RPA')
    #     st.write(len(df))
    #     st.write(df)

    ##########################################
    # Convert Date Index to Column (again)
    #df['Date'] = df.index
    df.reset_index(inplace=True)  # , drop=True
    # st.write('Full Original Dataset')
    # st.write(df)
    ###########################################
    # Till here df and ts (ts_) are equal
    # st.write(ts_)
    # st.write(idx_range_test)  # These are the ranges (index) for the Test data
    ###################################

    trainStartIndex = df.Date[df.Date ==
                              trainStartPeriod].index.values
    trainEndIndex = df.Date[df.Date ==
                            trainEndPeriod].index.values
    testStartIndex = df.Date[df.Date ==
                             testStartPeriod].index.values
    testEndIndex = df.Date[df.Date == testEndPeriod].index.values

    # st.write('trainStartIndex: ' + str(trainStartIndex))
    # st.write('trainEndIndex: ' + str(trainEndIndex))
    # st.write('testStartIndex: ' + str(testStartIndex))
    # st.write('testEndIndex: ' + str(testEndIndex))

    # st.write(type(trainStartIndex))

    idx_range_train = [int(trainStartIndex), int(trainEndIndex)]

    idx_range_test = [int(testStartIndex), int(testEndIndex)]

    df_train = df.iloc[idx_range_train[0]:idx_range_train[1] + 1, ]
    df_test = df.iloc[idx_range_test[0]:idx_range_test[1] + 1, ]

    with tab2:
        st.subheader(f'Stock {stockOptions}')
        with st.container():
            col1, col2 = st.columns([1, 3])
            with col1:
                tab21, tab22 = st.tabs(['Train', 'Test'])
                with tab21:
                    st.write(
                        'Total Number of Train Samples: ' + str(len(df_train)))
                    st.write('Start Train Date Index: ' +
                             str(int(trainStartIndex)))
                    st.write('End Train Date Index: ' +
                             str(int(trainEndIndex)))
                    st.dataframe(df_train)

                with tab22:
                    st.write(
                        'Total Number of Test Samples: ' + str(len(df_test)))
                    st.write('Start Test Date Index: ' +
                             str(int(testStartIndex)))
                    st.write('End Test Date Index: ' + str(int(testEndIndex)))
                    st.dataframe(df_test)

            with col2:
                #######################
                fig = go.Figure()
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(
                    go.Scatter(x=[trainStartPeriod, trainStartPeriod, trainEndPeriod, trainEndPeriod, trainStartPeriod],
                               y=[df[variableSelection].min(), df[variableSelection].max(),
                                  df[variableSelection].max(
                               ), df[variableSelection].min(),
                        df[variableSelection].min()], fill="toself", opacity=0.3,
                        mode="none", name=f"Train", fillcolor='green'))
                fig.add_trace(
                    go.Scatter(x=[testStartPeriod, testStartPeriod, testEndPeriod, testEndPeriod, testStartPeriod],
                               y=[df[variableSelection].min(), df[variableSelection].max(),
                                  df[variableSelection].max(
                               ), df[variableSelection].min(),
                        df[variableSelection].min()], fill="toself", opacity=0.3,
                        mode="none", name=f"Test", fillcolor='red'))
                fig.add_trace(
                    go.Scatter(x=df['Date'], y=df[variableSelection], name='Data', mode='lines', line=dict(color='blue')), secondary_y=False)
                fig.add_trace(
                    go.Scatter(x=df['Date'], y=df['norm'], name='Data (Norm)', mode='lines', line=dict(color='magenta')), secondary_y=True)
                fig.layout.update(
                    xaxis_rangeslider_visible=True)
                fig.update_layout(
                    autosize=False,
                    width=1000,
                    height=400,
                    plot_bgcolor="black",
                    margin=dict(
                        l=50,
                        r=50,
                        b=0,
                        t=0,
                        pad=2
                    ))
                # Set y-axes titles
                fig.update_yaxes(
                    title_text="Original", secondary_y=False)
                fig.update_yaxes(
                    title_text="Normalized", secondary_y=True)
                st.plotly_chart(fig)

    with tab3:
        st.json(data)

        # Closing file
        f.close()

    # init model
    model = model_from_cfg(cfg, ts_norm, scaler, idx_range_train)

    # fit model
    if not model.is_tf_model:
        print(f"\n-- Fitting {exp_name}... --")
        model.fit(ts_norm.values[idx_range_train[0] -
                                 model.max_lag:idx_range_train[1], :])
        model.save(save_path_curr)
    else:
        print(f"\n-- Fitting {exp_name} with {cfg.epochs} epochs... --")
        optimizer = tf.keras.optimizers.get(cfg.optimizer)
        K.set_value(optimizer.lr, cfg.learning_rate)

        # pass correct inp ts and prediction here
        ts_norm_in = model.get_inp_ts(
            ts_norm.values)  # Reshape to tensor format

        model.compile(optimizer=optimizer, loss=get_loss(
            cfg.loss), sample_weight_mode="temporal")

        callbacks = []
        callbacks.append(LRTensorBoard(
            log_dir=tb_path_curr, profile_batch=0))
        callbacks.append(MetricCallback(ts.to_numpy(), idx_range_train, idx_range_test, scaler, tb_path_curr,
                                        save_best_weights=cfg.save_best_weights))
        model.run_eagerly = cfg.run_eagerly

        if cfg.baseline_fit == 'WLS':  # Weighted Least Squared
            weights = 1 / \
                model(ts_norm_in[:, idx_range_train[0] -
                                 model.max_lag:idx_range_train[1] - 1, :])
        else:  # OLS case (Ordinary Least Squared)
            weights = tf.ones_like(
                ts_norm_in[:, idx_range_train[0]:idx_range_train[1], :])

        # st.write(weights)
        # st.write('Shape of Weight Tensor: ', tf.size(weights))
        # st.write(type(weights))

        valid_batch_gen_idxs = list(
            range(idx_range_train[0] + model.max_lag, idx_range_train[1] - cfg.label_length + 1))
        ds_fit = tf.data.Dataset.from_generator(
            model.random_batch_generator(ts_norm_in[:, :idx_range_train[1], :], idx_range_train,
                                         cfg.label_length,
                                         cfg.batch_size, cfg.steps_per_epoch,
                                         valid_batch_gen_idxs, weights),
            (tf.float32, tf.float32, tf.float32), output_shapes=(
                tf.TensorShape(
                    [cfg.batch_size, model.max_lag + cfg.label_length - 1,
                        model.channels_in]),
                tf.TensorShape(
                    [cfg.batch_size, cfg.label_length, model.channels_out]),
                tf.TensorShape(
                    [cfg.batch_size, cfg.label_length, model.channels_out])
            ))

        history = model.fit(ds_fit, epochs=cfg.epochs, verbose=cfg.verbose,
                            callbacks=callbacks)  # [TqdmCallback(verbose=1)]

        # plot optimization
        # dpv.plot_values(list(history.history.values()), None, fmt="-", logx=False,
        #                 title_str="Optimization History for Model %s" % config.model.name,
        #                 labels=list(history.history.keys()))
        # plt.gcf().savefig(Path(save_path_curr + "/optimization.pdf"))
        # model.summary()
        chkpt_model = tf.train.Checkpoint(model=model)
        chkpt_model.write(save_path_curr + "/model_params")
        pd.DataFrame.from_dict(history.history).to_csv(
            save_path_curr + '/metrics_history.csv', index=False)

    ts_train = ts.to_numpy()
    ts_train = ts_train[idx_range_train[0] -
                        model.max_lag:idx_range_train[1], :]
    ts_train_pred, ts_train_norm_pred, ts_train_norm_pred_raw, target_train, target_train_norm, target_train_norm_raw, pred_train_range = get_pred(model, scaler,
                                                                                                                                                   ts_train)

    ts_test = ts.to_numpy()
    ts_test = ts_test[idx_range_test[0] -
                      model.max_lag:idx_range_test[1], :]
    ts_test_pred, ts_test_norm_pred, ts_test_norm_pred_raw, target_test, target_test_norm, target_test_norm_raw, pred_test_range = get_pred(model, scaler,
                                                                                                                                            ts_test)

    metrics_train = get_model_metrics(
        model, scaler, ts.to_numpy(), idx_range_train, prefix='train')  #
    df_metrics_train = pd.DataFrame(metrics_train, index=[cfg.model])

    metrics_test = get_model_metrics(
        model, scaler, ts.to_numpy(), idx_range_test, prefix='test')  #
    df_metrics_test = pd.DataFrame(metrics_test, index=[cfg.model])
    df_metrics = pd.concat([df_metrics_train, df_metrics_test], axis=1)
    # print("")
    # print(df_metrics)
    df_metrics.to_csv(save_path_curr + "/metrics.csv")
    st.sidebar.success(
        f"Model Successfully Trained")
    st.balloons()
    ##############################
    metricsHistory = pd.read_csv('./results/config/metrics_history.csv')

    with tab4:
        tab41, tab42, tab43 = st.tabs(['Train', 'Test', 'Loss'])

        with tab41:
            st.dataframe(metricsHistory.iloc[:, :6])
        with tab42:
            st.dataframe(metricsHistory.iloc[:, 7:14])
        with tab43:
            st.dataframe(metricsHistory.iloc[:, 14])
    with tab5:

        tab51, tab52, tab53 = st.tabs(['Train', 'Test', 'Loss'])

        with tab51:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=metricsHistory.index, y=metricsHistory['train__MAE'], name='Train MAE'))
            fig.add_trace(
                go.Scatter(x=metricsHistory.index, y=metricsHistory['train__MSE'], name='Train MSE'))
            fig.add_trace(
                go.Scatter(x=metricsHistory.index, y=metricsHistory['train__QLIKE'], name='Train QLIKE'))
            fig.add_trace(
                go.Scatter(x=metricsHistory.index, y=metricsHistory['train__norm_MAE'], name='Train Norm MAE'))
            fig.add_trace(
                go.Scatter(x=metricsHistory.index, y=metricsHistory['train__norm_MSE'], name='Train Norm MSE'))
            fig.add_trace(
                go.Scatter(x=metricsHistory.index, y=metricsHistory['train__norm_QLIKE'], name='Train Norm QLIKE'))
            fig.add_trace(
                go.Scatter(x=metricsHistory.index, y=metricsHistory['train_loss'], name='Train Loss'))
            fig.layout.update(
                xaxis_rangeslider_visible=True)
            fig.update_layout(
                autosize=False,
                width=1400,
                height=400,
                plot_bgcolor="black",
                margin=dict(
                    l=50,
                    r=50,
                    b=0,
                    t=0,
                    pad=2
                ))
            st.plotly_chart(fig)

        with tab52:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=metricsHistory.index, y=metricsHistory['test__MAE'], name='Test MAE'))
            fig.add_trace(
                go.Scatter(x=metricsHistory.index, y=metricsHistory['test__MSE'], name='Test MSE'))
            fig.add_trace(
                go.Scatter(x=metricsHistory.index, y=metricsHistory['test__QLIKE'], name='Test QLIKE'))
            fig.add_trace(
                go.Scatter(x=metricsHistory.index, y=metricsHistory['test__norm_MAE'], name='Test Norm MAE'))
            fig.add_trace(
                go.Scatter(x=metricsHistory.index, y=metricsHistory['test__norm_MSE'], name='Test Norm MSE'))
            fig.add_trace(
                go.Scatter(x=metricsHistory.index, y=metricsHistory['test__norm_QLIKE'], name='Test Norm QLIKE'))
            fig.add_trace(
                go.Scatter(x=metricsHistory.index, y=metricsHistory['test_loss'], name='Test Loss'))
            fig.layout.update(
                xaxis_rangeslider_visible=True)
            fig.update_layout(
                autosize=False,
                width=1400,
                height=400,
                plot_bgcolor="black",
                margin=dict(
                    l=50,
                    r=50,
                    b=0,
                    t=0,
                    pad=2
                ))
            st.plotly_chart(fig)

        with tab53:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=metricsHistory.index, y=metricsHistory['loss'], name='Test Loss'))
            fig.layout.update(
                xaxis_rangeslider_visible=True)
            fig.update_layout(
                autosize=False,
                width=1400,
                height=400,
                plot_bgcolor="black",
                margin=dict(
                    l=50,
                    r=50,
                    b=0,
                    t=0,
                    pad=2
                ))

            st.plotly_chart(fig)

    #################################
    metrics = pd.read_csv('./results/config/metrics.csv')

    with tab6:
        st.dataframe(metrics)

    with tab7:
        col1, col2 = st.columns([1, 5])
        ts_norm.index = pd.to_datetime(ts_norm.index).date
        col1.write(ts_norm)

        ########################
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=ts_norm.index, y=ts_norm[0], name='Data'))
        fig.layout.update(
            xaxis_rangeslider_visible=True)
        fig.update_layout(
            autosize=False,
            width=1100,
            height=400,
            plot_bgcolor="black",
            margin=dict(
                l=50,
                r=50,
                b=0,
                t=0,
                pad=2
            ))
        col2.plotly_chart(fig)

    with tab8:
        tab81, tab82 = st.tabs(['Train', 'Test'])

        with tab81:
            col1, col2 = st.columns([1, 3])
            col1.write('Total Number of Train Samples: ' +
                       str(len(ts_train_pred)))
            col1.dataframe(ts_train_pred)
            # ts_train_pred, ts_train_norm_pred, ts_train_norm_pred_raw, target_train, target_train_norm, target_train_norm_raw, pred_train_range

            ########################
            # fig = go.Figure()
            # fig.add_trace(
            #     go.Scatter(x=ts_norm.index, y=ts_norm[0], name='Data'))
            # fig.layout.update(
            #     xaxis_rangeslider_visible=True)
            # fig.update_layout(
            #     autosize=False,
            #     width=900,
            #     height=400,
            #     plot_bgcolor="black",
            #     margin=dict(
            #         l=50,
            #         r=50,
            #         b=0,
            #         t=0,
            #         pad=2
            #     ))
            # col2.plotly_chart(fig)

        with tab82:
            col1, col2 = st.columns([1, 3])
            col1.write('Total Number of Test Samples: ' +
                       str(len(ts_test)))
            col1.dataframe(ts_test)

            col1.write('Total Number of Test Samples: ' +
                       str(len(ts_test_pred)))
            col1.dataframe(ts_test_pred)
            #ts_test_pred, ts_test_norm_pred, ts_test_norm_pred_raw, target_test, target_test_norm, target_test_norm_raw, pred_test_range

######################################
with st.sidebar.container():
    st.sidebar.subheader("University of Vienna | Research Project")
    st.sidebar.write(
        "###### App Authors: Roderick Perez & Le Thi (Janie) Thuy Trang")
    st.sidebar.write("###### Faculty Advisor: Xandro Bayer")
    st.sidebar.write("###### Updated: 12/10/2022")
