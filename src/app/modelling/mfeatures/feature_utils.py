import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, chi2_contingency

sys.path.insert(1, os.path.join(sys.path[0], '../../utils/'))
sys.path.insert(1, os.path.join(sys.path[0], '../../../config/'))
from db_connection import DB
from config import db_config


def bootstrap(
        data1,
        data2,
        n=5000,
        func=np.std,
        subtr=np.subtract,
        alpha=0.05):
    """
    Бутстрап средних значений для двух групп

    data1 - выборка 1 группы
    data2 - выборка 2 группы
    n=1000 - сколько раз моделировать
    func=np.mean - функция от выборки
    subtr=np.subtract,
    alpha=0.05 - 95% доверительный интервал

    return:
    ci_diff - доверительный интервал разницы средних для двух групп
    """
    s1, s2 = [], []
    s1_size = len(data1)
    s2_size = len(data2)

    for _ in range(n):
        itersample1 = np.random.choice(data1, size=s1_size, replace=True)
        s1.append(func(itersample1))
        itersample2 = np.random.choice(data2, size=s2_size, replace=True)
        s2.append(func(itersample2))
    s1.sort()
    s2.sort()

    # доверительный интервал разницы
    bootdiff = subtr(s2, s1)
    bootdiff.sort()

    ci_diff = (np.round(bootdiff[np.round(n * alpha / 2).astype(int)], 3),
               np.round(bootdiff[np.round(n * (1 - alpha / 2)).astype(int)], 3))

    return ci_diff


def verdict(ci_diff):
    cidiff_min = 0.001  # близкое к 0
    ci_diff_abs = [abs(ele) for ele in ci_diff]
    if min(ci_diff) <= cidiff_min <= max(ci_diff):
        return 0
    elif (cidiff_min >= max(ci_diff_abs) >= 0) or (cidiff_min >= min(ci_diff_abs) >= 0):
        return 0
    else:
        return 1


def importance_numeric(df: pd.DataFrame, features: list) -> list:
    """
    Выбираем значимые признаки на основе голосования бутстрепов и критерия Манна-Уитни

    :param df: датафрейм с данными;
    :param features: исходные признаки;
    :return: список значимых признаков
    """
    importance_col = []
    for feature in features:
        voices = []
        _, p_mw = mannwhitneyu(df[df['target'] == 0][feature], df[df['target'] == 1][feature])
        voices.append(1 if p_mw < 0.05 else 0)

        mean_diff = bootstrap(df[df['target'] == 0][feature], df[df['target'] == 1][feature], func=np.mean)
        voices.append(verdict(mean_diff))

        std_diff = bootstrap(df[df['target'] == 0][feature], df[df['target'] == 1][feature])
        voices.append(verdict(std_diff))

        if sum(voices) > 1:
            importance_col.append(feature)

    return importance_col


def importance_cat(df: pd.DataFrame, features: list, alpha=0.05) -> list:
    """
    Выбираем значимые признаки, с помощью критерия Хи квадрат

    :param df: датафрейм с данными;
    :param features: исходные признаки;
    :param alpha: уровень доверия;
    :return: список значимых признаков
    """
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


def features_selection(DATA_PATH: str, OUTPUT_PATH: str) -> None:
    """
    Обрабатываем признаки, делим фичи на количественные и категориальные,
    отбираем значимые и сохраняем датафрейм с этими признаками

    :param DATA_PATH: Путь к исходной таблице;
    :param OUTPUT_PATH: Путь для сохранения таблицы со значимыми признаками
    """
    df = merge_target(DATA_PATH)
    df_train = df.loc[df['target'].isin([0, 1])] \
        .drop(columns=['sk_id_curr', 'sk_id_prev'], errors='ignore') \
        .fillna(-999999)

    cat_features = ['flag_full_info_house', 'flag_delay_change_doc']

    if 'flag_full_info_house' in df_train.columns:
        numeric_features = [col for col in df_train.columns if col not in cat_features and col != 'target']
        meaningful_numeric = importance_numeric(df_train, numeric_features)
        meaningful_cat = importance_cat(df_train, cat_features)
        df[['sk_id_curr'] + meaningful_numeric + meaningful_cat].to_csv(OUTPUT_PATH, index=False)
    else:
        numeric_features = [col for col in df_train.columns if col != 'target']
        meaningful_numeric = importance_numeric(df_train, numeric_features)
        df[['sk_id_curr'] + meaningful_numeric].to_csv(OUTPUT_PATH, index=False)


def merge_target(DATA_PATH: str) -> pd.DataFrame:
    """
    Добавление таргета к признакам

    :param DATA_PATH: путь к таблицам с признаками;
    :return: итоговый датафрейм
    """
    data = pd.read_csv(DATA_PATH)
    db = DB(db_config)
    target = db.get_df_from_query("""SELECT sk_id_curr, target FROM application_train_test""")

    if 'credit_scoring' or 'pos_cash_balance' in DATA_PATH:
        data = data.groupby('sk_id_curr', as_index=False).mean()
    elif 'installments_payments' in DATA_PATH:
        data = data.groupby('sk_id_curr', as_index=False).sum()

    df = target.merge(data, on='sk_id_curr', how='left')
    return df
