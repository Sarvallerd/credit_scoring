import os
import sys
import pandas as pd

sys.path.insert(1, os.path.join(sys.path[0], '../../utils/'))
sys.path.insert(1, os.path.join(sys.path[0], '../../../config/'))

from db_connection import DB
from config import db_config

db = DB(db_config)

bureau_df = db.get_df_from_query("""SELECT * FROM bureau""")
bureau_balance_df = db.get_df_from_query("""SELECT * FROM bureau_balance""")


def bureau_balance_features(bureau_balance: pd.DataFrame) -> None:
    bureau_balance = bureau_balance.merge(bureau_df[['sk_id_bureau', 'sk_id_curr']], on='sk_id_bureau')

    features = pd.DataFrame(data=sorted(bureau_balance['sk_id_curr'].unique()), columns=['sk_id_curr'])

    # Кол-во открытых кредитов
    # Кол-во закрытых кредитов
    # Кол-во просроченных кредитов по разным дням просрочки
    pivot = pd.pivot_table(bureau_balance.loc[(bureau_balance['months_balance'] == -1)],
                           index='sk_id_curr',
                           columns='status',
                           values='sk_id_bureau',
                           aggfunc='count',
                           fill_value=0)\
        .rename(columns={'0': 'count_open_credits',
                         'C': 'count_closed_credits',
                         '1': 'count_1_overdue',
                         '2': 'count_2_overdue',
                         '3': 'count_3_overdue',
                         '4': 'count_4_overdue',
                         '5': 'count_5_overdue'})\
        .drop(columns='X')\
        .reset_index()
    features = features.merge(pivot, on='sk_id_curr', how='left').fillna(0)

    # Кол-во кредитов
    df_count_credits = bureau_balance \
        .groupby('sk_id_curr', as_index=False) \
        .agg(count_credits=pd.NamedAgg(column='sk_id_bureau', aggfunc='nunique'))
    features = features \
        .merge(df_count_credits, on='sk_id_curr', how='left')

    # Доля закрытых кредитов
    features['ratio_open_credits'] = features['count_open_credits'] / features['count_credits']

    # Доля открытых кредитов
    features['ratio_closed_credits'] = features['count_closed_credits'] / features['count_credits']

    # Доля просроченных кредитов по разным дням просрочки
    features['ratio_1_overdue'] = features['count_1_overdue'] / features['count_credits']
    features['ratio_2_overdue'] = features['count_2_overdue'] / features['count_credits']
    features['ratio_3_overdue'] = features['count_3_overdue'] / features['count_credits']
    features['ratio_4_overdue'] = features['count_4_overdue'] / features['count_credits']
    features['ratio_5_overdue'] = features['count_5_overdue'] / features['count_credits']

    # Интервал между последним закрытым кредитом и текущей заявкой
    df_last_close = bureau_balance[(bureau_balance['status'] == 'C')]\
        .groupby(['sk_id_curr', 'sk_id_bureau'], as_index=False)\
        .agg(month_close=pd.NamedAgg(column='months_balance', aggfunc='min'))
    df_last_close = df_last_close.groupby('sk_id_curr', as_index=False)['month_close'].max()
    df_last_close['month_close'] = df_last_close['month_close'].apply(lambda x: abs(x))
    features = features.merge(df_last_close, on='sk_id_curr', how='left')

    # Интервал между взятием последнего активного займа и текущей заявкой
    df_last_active = bureau_balance.loc[~(bureau_balance['status'].isin(['C', 'X']))] \
        .groupby(['sk_id_curr', 'sk_id_bureau'], as_index=False) \
        .agg(month_active=pd.NamedAgg(column='months_balance', aggfunc='min'))
    df_last_active = df_last_active.groupby('sk_id_curr', as_index=False)['month_active'].max()
    df_last_active['month_active'] = df_last_active['month_active'].apply(lambda x: abs(x))
    features = features.merge(df_last_active, on='sk_id_curr', how='left')

    features.set_index('sk_id_curr', inplace=True)
    features.to_csv('features_csv/bureau_balance_features.csv')


def bureau_features(bureau: pd.DataFrame) -> None:
    features = pd.DataFrame(index=sorted(bureau['sk_id_curr'].unique()))

    # Максимальная сумма просрочки
    features['max_amt_overdue'] = bureau.groupby('sk_id_curr')['amt_credit_sum_overdue'].max()

    # Минимальная сумма просрочки
    features['min_amt_overdue'] = bureau.groupby('sk_id_curr')['amt_credit_sum_overdue'].min()

    # Какую долю суммы от открытого займа просрочил
    df_active_credits = bureau.loc[bureau['credit_active'] != 'Closed'].groupby('sk_id_curr', as_index=False).agg(
        sum_amt=pd.NamedAgg(column='amt_credit_sum', aggfunc='sum'),
        sum_amt_ovredue=pd.NamedAgg(column='amt_credit_sum_overdue', aggfunc='sum')
    )
    df_active_credits['ratio_active_credits'] = df_active_credits['sum_amt_ovredue'] / df_active_credits['sum_amt']
    features = features.merge(df_active_credits['ratio_active_credits'], right_index=True, left_index=True, how='left')

    credit_types = bureau['credit_type'].unique()

    # Кол-во кредитов определенного типа
    pivot = pd.pivot_table(
        bureau,
        index='sk_id_curr',
        columns='credit_type',
        values='sk_id_bureau',
        aggfunc='count',
        fill_value=0
    ).rename(columns={cr_tp: f'count_{cr_tp}' for cr_tp in credit_types})
    features = pd.concat([features, pivot], axis=1)

    # Кол-во просрочек кредитов определенного типа
    pivot = pd.pivot_table(
        bureau.loc[bureau['credit_day_overdue'] != 0],
        index='sk_id_curr',
        columns='credit_type',
        values='sk_id_bureau',
        aggfunc='count',
        fill_value=0
    ).rename(columns={cr_tp: f'count_overdue_{cr_tp}' for cr_tp in credit_types})
    features = features.merge(pivot, how='left', left_index=True, right_index=True).fillna(0)

    # Кол-во закрытых кредитов определенного типа
    pivot = pd.pivot_table(
        bureau.loc[bureau['credit_active'] == 'Closed'],
        index='sk_id_curr',
        columns='credit_type',
        values='sk_id_bureau',
        aggfunc='count',
        fill_value=0
    ).rename(columns={cr_tp: f'count_closed_{cr_tp}' for cr_tp in credit_types})
    features = features.merge(pivot, how='left', left_index=True, right_index=True).fillna(0)

    features.to_csv('features_csv/bureau_features.csv', index_label='sk_id_curr')


bureau_balance_features(bureau_balance_df)
bureau_features(bureau_df)