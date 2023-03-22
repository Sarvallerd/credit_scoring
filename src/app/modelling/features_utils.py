import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, chi2_contingency

sys.path.insert(1, os.path.join(sys.path[0], '../utils/'))
sys.path.insert(1, os.path.join(sys.path[0], '../../config/'))
from db_connection import DB
from config import db_config


def merge_data(DATA_PATH: str, files_names: list) -> pd.DataFrame:
    """Склеивание всех признаков в один датафрейм."""
    app = pd.read_csv(DATA_PATH + files_names[0])
    db = DB(db_config)
    target = db.get_df_from_query("""SELECT sk_id_curr, target FROM application_train_test""")
    bureau = pd.read_csv(DATA_PATH + files_names[1])
    bureau_balance = pd.read_csv(DATA_PATH + files_names[2])
    credit_card = pd.read_csv(DATA_PATH + files_names[3])
    installments_payments = pd.read_csv(DATA_PATH + files_names[4])
    pos_cash_balance = pd.read_csv(DATA_PATH + files_names[5])
    previous_application = pd.read_csv(DATA_PATH + files_names[6])

    df = app.merge(target, on='sk_id_curr')
    df = df.merge(bureau, on='sk_id_curr', how='left')
    df = df.merge(bureau_balance, on='sk_id_curr', how='left')
    df = df.merge(previous_application, on='sk_id_curr', how='left')

    pos_cash_balance = pos_cash_balance.groupby('sk_id_curr', as_index=False).mean()
    df = df.merge(pos_cash_balance.drop(columns='sk_id_prev'), on='sk_id_curr', how='left')

    installments_payments = installments_payments.groupby('sk_id_curr', as_index=False).agg(
        {'num_instalment_number': 'sum',
         'flag_overdue': 'sum',
         'flag_overdue_amt': 'sum'})
    df = df.merge(installments_payments, on='sk_id_curr', how='left')

    credit_card = credit_card.groupby('sk_id_curr', as_index=False).mean()
    df = df.merge(credit_card.drop(columns='sk_id_prev'), on='sk_id_curr', how='left')

    return df


def importance_numeric(df: pd.DataFrame, features: list, alpha=0.05) -> list:
    """Применяем критерий Манна-Уитни для количественных признаков."""
    importance_col = []
    for feature in features:
        _, p_mw = mannwhitneyu(df[df['target'] == 0][feature], df[df['target'] == 1][feature])
        if p_mw < alpha:
            importance_col.append(feature)

    return importance_col


def importance_cat(df: pd.DataFrame, features: list, alpha=0.05) -> list:
    """Применяем критерий хи-квалрат для категориальных признаков."""
    importance_col = []
    for feature in features:
        cross_tab = pd.concat([
            pd.crosstab(df[feature], df['target'], margins=False),
            df.groupby(feature)['target'].agg(['count', 'mean']).round(4)
        ], axis=1)

        cross_tab['mean'] = np.round(cross_tab['mean'] * 100, 2)

        _, p, _, _ = chi2_contingency(cross_tab.values)

        if p < alpha:
            importance_col.append(feature)

    return importance_col


def features_selection(DATA_PATH: str, files_names: list) -> tuple[pd.DataFrame, list]:
    """Отбор значимых признаков."""
    df = merge_data(DATA_PATH, files_names)
    df_train = df.loc[df['target'].isin([0, 1])].drop(columns='sk_id_curr').fillna(-999999)
    df_train['ratio_active_credits'] = df_train['ratio_active_credits'].apply(lambda x: 0 if x == float('inf') else x)

    cat_features = ['flag_full_info_house', 'flag_delay_change_doc']
    numeric_features = [col for col in df_train.columns if col not in cat_features and col != 'target']

    meaningful_numeric = importance_numeric(df_train, numeric_features)
    meaningful_cat = importance_cat(df_train, cat_features)

    return df_train[meaningful_numeric + meaningful_cat + ['target']], meaningful_numeric
