import pandas as pd
import numpy as np
import sys
import random
import gc
from sklearn import linear_model, ensemble, svm, naive_bayes


def read_sample_csv(filename, sample_size=100000, **kwargs):
    nrows = sum(1 for line in open(filename))-1
    print("we got %s rows in %s" % (nrows, filename))
    if nrows < sample_size:
        print("reading everything")
        return pd.read_csv(filename, **kwargs)
    else:
        print("sampling %s rows" % sample_size)
        skip = sorted(random.sample(xrange(1, nrows+1), nrows-sample_size))
        return pd.read_csv(filename, skiprows=skip, **kwargs)

SAMPLING = 'sampling' in sys.argv
read_csv = read_sample_csv if SAMPLING else pd.read_csv
print("Sampling: %s" % SAMPLING)

types = {
    'Semana': np.uint8,
    'Agencia_ID': np.uint16,
    'Canal_ID': np.uint8,
    'Ruta_SAK': np.uint16,
    'Cliente_ID': np.uint32,
    'Producto_ID': np.uint32,
    'Demanda_uni_equil': np.uint32
}

train = read_csv('./input/train.csv', usecols=types.keys(), dtype=types)

types['id'] = np.uint32
del types['Demanda_uni_equil']
test = read_csv('./input/test.csv', usecols=types.keys(), dtype=types)

print("finish reading")

train['test'] = 0
test['test'] = 1
data = pd.concat([train, test])
del train
del test
gc.collect()

data.rename(columns={
        "Agencia_ID": "agency_id",
        "Canal_ID": "canal_id",
        "Cliente_ID": "client_id",
        "Producto_ID": "product_id",
        "Ruta_SAK": "route_id",
        "Semana": "week",
        "Demanda_uni_equil": "demand"
        }, inplace=True)


def add_lagging_demand_features(data, max_week=6, inplace=False):
    if inplace:
        data = data.copy()

    lag_cols = []
    demand_per_CPW = data\
        .groupby(["client_id", "product_id", "week"])["demand"]\
        .sum()

    for i in range(1, max_week+1):

        def get_lagging_demand(row):
            client_id, product_id = row["client_id"], row["product_id"]
            week = row["week"] - i

            if week < 3:
                # we don't have anything before week 3
                return pd.np.NaN

            if (client_id, product_id, week) in demand_per_CPW.index:
                return demand_per_CPW.loc[client_id, product_id, week]
            else:
                # we assume we got 0 demand for this product, client
                return 0

        col_name = "demand_lag%s" % i
        data[col_name] = data.apply(get_lagging_demand, axis=1)
        lag_cols.append(col_name)

    data["demand_lagTotal"] = 0
    for col in lag_cols:
        data["demand_lagTotal"] += data[col]
    lag_cols.append("demand_lagTotal")

    # # try to make a bit of space
    # del demand_per_CPW
    # gc.collect()

    return data, lag_cols


def add_frequency_features(data, inplace=False):
    if inplace:
        data = data.copy()

    freq_cols = []

    for aggr_level in ["agency", "route", "client", "product"]:
        aggr_col = aggr_level+"_id"
        new_col_name = aggr_level+"_demand"

        demand_per_aggr = data\
            .groupby(aggr_col)["demand"]\
            .mean().rename(new_col_name)\
            .reset_index()
        data = pd.merge(
            data, demand_per_aggr,
            on=aggr_level+"_id",
            how='left'
        )
        data[new_col_name].fillna(0, inplace=True)
        freq_cols.append(new_col_name)

        # try to make a bit of space
        del demand_per_aggr
        gc.collect()

    return data, freq_cols


def add_frequency_lagging_features(data, inplace=True, max_week=6):
    if inplace:
        data = data.copy()

    freq_lag_cols = []

    for aggr_level in ["agency", "route", "client", "product"]:
        aggr_col = aggr_level+"_id"
        new_col_name = aggr_level+"_demand"

        demand_per_aggr_and_week = data\
            .groupby([aggr_col, 'week'])["demand"]\
            .mean()\
            .rename(new_col_name)\
            .fillna(0)
        lag_cols = []
        for i in range(1, max_week+1):

            def get_lagging_demand(row):
                aggr_index = row[aggr_col]
                week = row["week"] - i

                if week < 3:
                    # we don't have anything before week 3
                    return pd.np.NaN

                if (aggr_index, week) in demand_per_aggr_and_week.index:
                    return demand_per_aggr_and_week.loc[aggr_index, week]
                else:
                    # we assume we got 0 demand for this product, client
                    return 0

            col_name = "%s_lag%s" % (new_col_name, i)
            data[col_name] = data.apply(get_lagging_demand, axis=1)

            lag_cols.append(col_name)
            freq_lag_cols.append(col_name)

        lagTotal_col = "%s_lagTotal" % new_col_name
        data[lagTotal_col] = 0
        for col in lag_cols:
            data[lagTotal_col] += data[col]
        freq_lag_cols.append(lagTotal_col)

        # # try to make a bit of space
        # del demand_per_aggr_and_week
        # gc.collect()

    return data, freq_lag_cols


def extract_features(data, inplace=False, max_lagging_shift=6):
    print("Start extracting features")
    features_columns = [
        'agency_id',
        'canal_id',
        'client_id',
        'product_id',
        'route_id'
    ]

    # add lagging columns features
    data, lag_cols = add_lagging_demand_features(
        data,
        inplace=inplace,
        max_week=max_lagging_shift
    )
    features_columns += lag_cols
    print("Finish lagging features")

    # add frequency features
    data, freq_cols = add_frequency_features(data, inplace=inplace)
    features_columns += freq_cols
    print("Finish frequency features")

    # add frequency lagging features
    data, freq_lag_cols = add_frequency_lagging_features(
        data,
        inplace=inplace,
        max_week=max_lagging_shift
    )
    features_columns += freq_lag_cols
    print("Finish frequency lagging features")

    return data, features_columns


def train_test_split(data, target_week=10, weeks_used=[7, 8, 9]):
    train = data[data.week.isin(weeks_used)]
    test = data[data.week == target_week]
    return train, test


print("let's predict week 10")
prep_data10, features_columns10 = extract_features(
    data,
    inplace=True,
    max_lagging_shift=3
)
prep_data10["log_demand"] = np.log1p(prep_data10["demand"])
train10, test10 = train_test_split(
    prep_data10,
    target_week=10,
    weeks_used=[6, 7, 8, 9]
)

# try to make some space
del prep_data10
gc.collect()

X_train10 = train10[features_columns10]
X_test10 = test10[features_columns10]
y_train10 = train10["log_demand"]

del train10
gc.collect()

clf10 = ensemble.RandomForestRegressor(n_jobs=-1)
print("fitting")
clf10.fit(X_train10, y_train10)

del X_train10, y_train10
gc.collect()

print("predict")
y_pred10 = clf10.predict(X_test10)
test10['demand_pred_10'] = np.exp(y_pred10)-1

del X_test10
gc.collect()

# add the prediction for week 10 to the data
# (data2 = data with prdiction of week 10)
data = pd.merge(data, test10[['id', 'demand_pred_10']], on='id', how='left')
data['demand'] = data.apply(
    lambda row: row['demand_pred_10'] if row['week'] == 10 else row['demand'],
    axis=1
)
del test10
gc.collect()


# In[ ]:
print("let's predict week 11")
# let's now go over again and predict week 11 with the data
# with have now (ie with week 10)
prep_data11, features_columns11 = extract_features(
    data,
    inplace=True,
    max_lagging_shift=4
)
prep_data11["log_demand"] = np.log1p(prep_data11["demand"])
train11, test11 = train_test_split(
    prep_data11,
    target_week=11,
    weeks_used=[7, 8, 9, 10]
)

del prep_data11
gc.collect()

X_train11 = train11[features_columns11]
X_test11 = test11[features_columns11]
y_train11 = train11["log_demand"]

del train11
gc.collect()

clf11 = ensemble.RandomForestRegressor(n_jobs=-1)
print("fitting")
clf11.fit(X_train11, y_train11)

del X_train11, y_train11
gc.collect()

print("predict")
y_pred11 = clf11.predict(X_test11)
test11['demand_pred_11'] = np.exp(y_pred11)-1

del X_test11, y_pred11
gc.collect()

# add the prediction for week 11 to the data
# (data3 = data with prdiction of week and 11)
data = pd.merge(data, test11[['id', 'demand_pred_11']], on='id', how='left')
data['demand'] = data.apply(
    lambda row: row['demand_pred_11'] if row['week'] == 11 else row['demand'],
    axis=1
)

del test11
gc.collect()

# In[ ]:
print("output submission")
submission = data[data.test == 1][['id', 'demand']]

del data
gc.collect()

submission.rename(columns={"demand": "Demanda_uni_equil"}, inplace=True)
submission.to_csv('./output/submission.csv', index=False)
