import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer


def preprocess(train, test1, test2, missing):
    """
    train:   train dataframe
    test1:   test dataframe (for evaluating model)
    test2:   test dataframe (for leaderboard)
    missing: 0 for simply dropping features with missing values, 1 for including missing indicators,
             2 for simple imputation (mode) (+ indicators), 3 for imputation based on age (+ indicators)
    """

    current_year = int(datetime.datetime.now().date().strftime("%Y"))
    train['customer_age'] = train.apply(
        lambda row: int(current_year - int(row.customer_birth_date.split('-')[0])), axis=1)
    test1['customer_age'] = test1.apply(
        lambda row: int(current_year - int(row.customer_birth_date.split('-')[0])), axis=1)
    test2['customer_age'] = test2.apply(
        lambda row: int(current_year - int(row.customer_birth_date.split('-')[0])), axis=1)

    if missing == 1:
        missing_features = ['customer_since_all', 'customer_since_bank', 'customer_occupation_code',
                            'customer_education', 'customer_children', 'customer_relationship']
        is_feature_missing = [f + '_missing' for f in missing_features]
        train[is_feature_missing] = train[missing_features].apply(lambda row: row.isna(), axis=1)
        test1[is_feature_missing] = test1[missing_features].apply(lambda row: row.isna(), axis=1)
        test2[is_feature_missing] = test2[missing_features].apply(lambda row: row.isna(), axis=1)

    elif missing == 2:
        missing_features = ['customer_since_all', 'customer_since_bank', 'customer_occupation_code',
                            'customer_education', 'customer_children', 'customer_relationship']
        is_feature_missing = [f + '_missing' for f in missing_features]
        train[is_feature_missing] = train[missing_features].apply(lambda row: row.isna(), axis=1)
        test1[is_feature_missing] = test1[missing_features].apply(lambda row: row.isna(), axis=1)
        test2[is_feature_missing] = test2[missing_features].apply(lambda row: row.isna(), axis=1)

        train['customer_since_all_years'] = train.apply(
            lambda row: int(current_year - int(row.customer_since_all.split('-')[0])) if (
                not pd.isna(row.customer_since_all)) else pd.NA, axis=1)
        train['customer_since_bank_years'] = train.apply(
            lambda row: int(current_year - int(row.customer_since_bank.split('-')[0])) if (
                not pd.isna(row.customer_since_bank)) else pd.NA, axis=1)

        test1['customer_since_all_years'] = test1.apply(
            lambda row: int(current_year - int(row.customer_since_all.split('-')[0])) if (
                not pd.isna(row.customer_since_all)) else pd.NA, axis=1)
        test1['customer_since_bank_years'] = test1.apply(
            lambda row: int(current_year - int(row.customer_since_bank.split('-')[0])) if (
                not pd.isna(row.customer_since_bank)) else pd.NA, axis=1)
        test2['customer_since_all_years'] = test2.apply(
            lambda row: int(current_year - int(row.customer_since_all.split('-')[0])) if (
                not pd.isna(row.customer_since_all)) else pd.NA, axis=1)
        test2['customer_since_bank_years'] = test2.apply(
            lambda row: int(current_year - int(row.customer_since_bank.split('-')[0])) if (
                not pd.isna(row.customer_since_bank)) else pd.NA, axis=1)

        train.pop('customer_since_all')
        train.pop('customer_since_bank')
        test1.pop('customer_since_all')
        test1.pop('customer_since_bank')
        test2.pop('customer_since_all')
        test2.pop('customer_since_bank')

        imp_values = {'customer_occupation_code': train['customer_occupation_code'].mode().values[0],
                      'customer_education': train['customer_education'].mode().values[0],
                      'customer_relationship': train['customer_relationship'].mode().values[0],
                      'customer_since_all_years': train['customer_since_all_years'].mode().values[0],
                      'customer_since_bank_years': train['customer_since_bank_years'].mode().values[0],
                      'customer_children': train['customer_children'].mode().values[0]}

        train = train.fillna(value=imp_values)
        test1 = test1.fillna(value=imp_values)
        test2 = test2.fillna(value=imp_values)

        str_col = ['customer_children', 'customer_relationship', 'customer_education', 'customer_occupation_code']
        train[str_col] = train[str_col].astype('string')
        test1[str_col] = test1[str_col].astype('string')
        test2[str_col] = test2[str_col].astype('string')

        int_col = ['customer_since_all_years', 'customer_since_bank_years']
        train[int_col] = train[int_col].astype('int64')
        test1[int_col] = test1[int_col].astype('int64')
        test2[int_col] = test2[int_col].astype('int64')

    elif missing == 3:
        missing_features = ['customer_since_all', 'customer_since_bank', 'customer_occupation_code',
                            'customer_education', 'customer_children', 'customer_relationship']
        is_feature_missing = [f + '_missing' for f in missing_features]
        train[is_feature_missing] = train[missing_features].apply(lambda row: row.isna(), axis=1)
        test1[is_feature_missing] = test1[missing_features].apply(lambda row: row.isna(), axis=1)
        test2[is_feature_missing] = test2[missing_features].apply(lambda row: row.isna(), axis=1)

        train['customer_since_all_years'] = train.apply(
            lambda row: int(current_year - int(row.customer_since_all.split('-')[0])) if (
                not pd.isna(row.customer_since_all)) else pd.NA, axis=1)
        train['customer_since_bank_years'] = train.apply(
            lambda row: int(current_year - int(row.customer_since_bank.split('-')[0])) if (
                not pd.isna(row.customer_since_bank)) else pd.NA, axis=1)

        test1['customer_since_all_years'] = test1.apply(
            lambda row: int(current_year - int(row.customer_since_all.split('-')[0])) if (
                not pd.isna(row.customer_since_all)) else pd.NA, axis=1)
        test1['customer_since_bank_years'] = test1.apply(
            lambda row: int(current_year - int(row.customer_since_bank.split('-')[0])) if (
                not pd.isna(row.customer_since_bank)) else pd.NA, axis=1)
        test2['customer_since_all_years'] = test2.apply(
            lambda row: int(current_year - int(row.customer_since_all.split('-')[0])) if (
                not pd.isna(row.customer_since_all)) else pd.NA, axis=1)
        test2['customer_since_bank_years'] = test2.apply(
            lambda row: int(current_year - int(row.customer_since_bank.split('-')[0])) if (
                not pd.isna(row.customer_since_bank)) else pd.NA, axis=1)

        train.pop('customer_since_all')
        train.pop('customer_since_bank')
        test1.pop('customer_since_all')
        test1.pop('customer_since_bank')
        test2.pop('customer_since_all')
        test2.pop('customer_since_bank')

        impute_values_kids = get_impute_values(train, 'customer_children', 'string')
        train['customer_children'] = train.apply(
            lambda row: impute(row, impute_values_kids, 'customer_children'), axis=1)
        test1['customer_children'] = test1.apply(
            lambda row: impute(row, impute_values_kids, 'customer_children'), axis=1)
        test2['customer_children'] = test2.apply(
            lambda row: impute(row, impute_values_kids, 'customer_children'), axis=1)

        impute_values_all = get_impute_values(train, 'customer_since_all_years', 'int64')
        train['customer_since_all_years'] = train.apply(
            lambda row: impute(row, impute_values_all, 'customer_since_all_years'), axis=1)
        test1['customer_since_all_years'] = test1.apply(
            lambda row: impute(row, impute_values_all, 'customer_since_all_years'), axis=1)
        test2['customer_since_all_years'] = test2.apply(
            lambda row: impute(row, impute_values_all, 'customer_since_all_years'), axis=1)

        impute_values_bank = get_impute_values(train, 'customer_since_bank_years', 'int64')
        train['customer_since_bank_years'] = train.apply(
            lambda row: impute(row, impute_values_bank, 'customer_since_bank_years'), axis=1)
        test1['customer_since_bank_years'] = test1.apply(
            lambda row: impute(row, impute_values_bank, 'customer_since_bank_years'), axis=1)
        test2['customer_since_bank_years'] = test2.apply(
            lambda row: impute(row, impute_values_bank, 'customer_since_bank_years'), axis=1)

        imp_values = {'customer_occupation_code': train['customer_occupation_code'].mode().values[0],
                      'customer_education': train['customer_education'].mode().values[0],
                      'customer_relationship': train['customer_relationship'].mode().values[0]}
        train = train.fillna(value=imp_values)
        test1 = test1.fillna(value=imp_values)
        test2 = test2.fillna(value=imp_values)

        str_col = ['customer_children', 'customer_relationship', 'customer_education', 'customer_occupation_code']
        train[str_col] = train[str_col].astype('string')
        test1[str_col] = test1[str_col].astype('string')
        test2[str_col] = test2[str_col].astype('string')

        int_col = ['customer_since_all_years', 'customer_since_bank_years']
        train[int_col] = train[int_col].astype('int64')
        test1[int_col] = test1[int_col].astype('int64')
        test2[int_col] = test2[int_col].astype('int64')

    train.dropna(axis=1, inplace=True)
    test1.dropna(axis=1, inplace=True)
    test2.dropna(axis=1, inplace=True)

    str_cols = ['client_id', 'customer_birth_date', 'customer_postal_code']
    train[str_cols] = train[str_cols].astype('string')
    test1[str_cols] = test1[str_cols].astype('string')
    test2[str_cols] = test2[str_cols].astype('string')

    train[['visits_distinct_so', 'visits_distinct_so_areas']] = train[['visits_distinct_so',
                                                                       'visits_distinct_so_areas']].astype('int64')
    test1[['visits_distinct_so', 'visits_distinct_so_areas']] = test1[['visits_distinct_so',
                                                                       'visits_distinct_so_areas']].astype('int64')
    test2[['visits_distinct_so', 'visits_distinct_so_areas']] = test2[['visits_distinct_so',
                                                                       'visits_distinct_so_areas']].astype('int64')

    train['customer_gender'] = train['customer_gender'].apply(lambda row: row - 1)
    test1['customer_gender'] = test1['customer_gender'].apply(lambda row: row - 1)
    test2['customer_gender'] = test2['customer_gender'].apply(lambda row: row - 1)

    bool_columns = train.select_dtypes(include='bool').columns.tolist()
    train[bool_columns] = train[bool_columns].astype('int64')
    test1[bool_columns] = test1[bool_columns].astype('int64')
    test2[bool_columns] = test2[bool_columns].astype('int64')

    # train = train[train['customer_age'] <= 100]

    train.pop('customer_birth_date')
    test1.pop('customer_birth_date')
    test2.pop('customer_birth_date')

    train['customer_postal_code'] = train['customer_postal_code'].apply(lambda row: row[0])
    test1['customer_postal_code'] = test1['customer_postal_code'].apply(lambda row: row[0])
    test2['customer_postal_code'] = test2['customer_postal_code'].apply(lambda row: row[0])

    non_num_columns = list(set(train.select_dtypes(exclude='number').columns.tolist()) - {"client_id"})
    transformer = make_column_transformer(
        (OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore'), non_num_columns),
        remainder='passthrough')

    transformed_train = transformer.fit_transform(train)
    new_col_names = transformer.get_feature_names_out()

    transformed_test1 = transformer.transform(test1)
    transformed_test2 = transformer.transform(test2)

    train = pd.DataFrame(transformed_train, columns=new_col_names)
    test1 = pd.DataFrame(transformed_test1, columns=new_col_names)
    test2 = pd.DataFrame(transformed_test2, columns=new_col_names)

    return [train, test1, test2]


def get_impute_values(df, feature, f_type):
    impute_df = df[['customer_age', feature]].copy()
    impute_df.dropna(axis=0, inplace=True)
    impute_df[feature] = impute_df[feature].astype(f_type)
    impute_values = impute_df.groupby('customer_age')[feature].agg(lambda x: pd.Series.mode(x)[0]).to_frame()
    return impute_values


def impute(row, impute_values, feature):
    if pd.isna(row[feature]):
        if row['customer_age'] in impute_values.index:
            return impute_values.loc[row['customer_age']].values[0]
        else:
            lower = impute_values.index[impute_values.index < row['customer_age']]
            higher = impute_values.index[impute_values.index > row['customer_age']]
            lo = 500
            hi = 500
            closest_age = 0
            if len(lower) > 0:
                lo = lower[-1]
                closest_age = lo
            if len(higher) > 0:
                hi = higher[0]
            if np.abs(row['customer_age'] - hi) < np.abs(row['customer_age'] - lo):
                closest_age = hi

            return impute_values.loc[closest_age].values[0]
    else:
        return row[feature]
