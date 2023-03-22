import os
import sys
import pandas as pd

sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
sys.path.insert(1, os.path.join(sys.path[0], '../../config/'))
from db_connection import DB
from config import db_config


def rule_interest_rate(df: pd.DataFrame, lb=0.022, ub=0.023) -> set:
    """Условие на процентную ставку."""
    cond = (df['interest_rate'] <= ub) & (df['interest_rate'] > lb)
    df = df.loc[cond]

    return set(df['sk_id_curr'])


def rule_ext_scores(df: pd.DataFrame, lb=0, ub=0.01) -> set:
    """Условие на рейтинг внешних источников."""
    cond = (df['weighted_ext_scores'] <= ub) & (df['weighted_ext_scores'] > lb)
    df = df.loc[cond]

    return set(df['sk_id_curr'])


def rule_amt_ratio(bound=14) -> set:
    """Условие на долю суммы заема к сумме дохода."""
    db = DB(db_config)
    query = """SELECT sk_id_curr, amt_income_total, amt_credit FROM application_train_test"""
    df = db.get_df_from_query(query)
    df['amt_ratio'] = df['amt_credit'] / df['amt_income_total']
    cond = (df['amt_ratio'] > bound)
    df = df.loc[cond]

    return set(df['sk_id_curr'])


def filter_clients(df: pd.DataFrame) -> pd.DataFrame:
    """Избавляемся от клиентов, которые попадают под правила антифрода."""
    interest_rate_ids = rule_interest_rate(df)
    ext_scores_ids = rule_ext_scores(df)
    amt_ratio_ids = rule_amt_ratio()

    fraudster_ids = interest_rate_ids | ext_scores_ids | amt_ratio_ids

    return df.loc[~(df['sk_id_curr'].isin(fraudster_ids))]
