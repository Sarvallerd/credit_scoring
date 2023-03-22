import os
import sys
import pickle
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, chi2_contingency
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from catboost import CatBoostClassifier

sys.path.insert(1, os.path.join(sys.path[0], '../app/utils'))
sys.path.insert(1, os.path.join(sys.path[0], '../config/'))
from db_connection import DB
from config import db_config


LOGREG_PATH = "D:\\PycharmProjects\\credit_scoring\\src\\hw_6\\models\\logreg.pkl"
DTC_PATH = "D:\\PycharmProjects\\credit_scoring\\src\\hw_6\\models\\dtc.pkl"
CBC_PATH = "D:\\PycharmProjects\\credit_scoring\\src\\hw_6\\models\\cbc.pkl"
RFC_PATH = "D:\\PycharmProjects\\credit_scoring\\src\\hw_6\\models\\rfc.pkl"
DATA_PATH = "D:\\PycharmProjects\\credit_scoring\\src\\app\\modelling\\features\\features_csv\\"
files_names = ["application_features.csv", "bureau_balance_features.csv", "bureau_features.csv",
               "credit_card_features.csv", "installments_payments_features.csv", "pos_cash_balance.csv",
               "previous_application_features.csv"]


def merge_data(DATA_PATH: str, files_names: str) -> pd.DataFrame:
    """
    Склеивание всех признаков в один датафрейм

    :param DATA_PATH: путь к таблицам с признаками;
    :param files_names: список с названиями таблиц;
    :return: итоговый датафрейм
    """
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
    importance_col = []
    for feature in features:
        _, p_mw = mannwhitneyu(df[df['target'] == 0][feature], df[df['target'] == 1][feature])
        if p_mw < alpha:
            importance_col.append(feature)

    return importance_col


def importance_cat(df: pd.DataFrame, features: list, alpha=0.05) -> list:
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


def features_selection(df: pd.DataFrame) -> tuple[list, list]:
    """
    Поиск значимых признаков

    :param df: датафрейм с данными;
    :return: списки со значимыми количественными и категориальными признаками
    """
    cat_features = ['flag_full_info_house', 'flag_delay_change_doc']
    numeric_features = [col for col in df.columns if col not in cat_features and col != 'target']

    meaningful_numeric = importance_numeric(df, numeric_features)
    meaningful_cat = importance_cat(df, cat_features)

    return meaningful_numeric, meaningful_cat


def create_dataframe(DATA_PATH: str, files_names: list) -> pd.DataFrame:
    """
    Склеиваем посчитанные признаки, выбираем занчимые, сохраняем итоговый датафрейм

    :param DATA_PATH: путь к таблицам с признаками;
    :param files_names: список с названиями таблиц;
    :return: Итоговый датафрейм
    """
    df_all = merge_data(DATA_PATH, files_names)
    df_train = df_all.loc[df_all['target'].isin([0, 1])].drop(columns='sk_id_curr').fillna(-999999)
    df_train['ratio_active_credits'] = df_train['ratio_active_credits'].apply(lambda x: 0 if x == float('inf') else x)

    meaningful_numeric, meaningful_cat = features_selection(df_train)

    df_all = df_all[meaningful_numeric + meaningful_cat + ['target']].fillna(-999999)
    df_all['ratio_active_credits'] = df_all['ratio_active_credits'].apply(lambda x: 0 if x == float('inf') else x)

    return df_all


def cbc_tuning(CBC_PATH, X_train, X_val, y_train, y_val, cv):
    """
    Подбор гиперпараметров и обучение модели с лучшими гиперпараметрами

    :param CBC_PATH: путь для сохранения модели;
    :param X_train: выборка для обучения;
    :param X_val: выборка для валидации;
    :param y_train: таргет для обучения;
    :param y_val: таргет для валидации;
    :param cv: алгоритм кросс-валидации;
    :return: обученная модели, скор на выборке для валидации
    """
    cbc = CatBoostClassifier(random_state=22, verbose=False)

    params_cbc = {
        'learning_rate': [0.01, 0.001],
        'iterations': [100, 200],
        'depth': [5, 10]
    }
    gs_cbc = GridSearchCV(cbc, param_grid=params_cbc, cv=cv, scoring='roc_auc')
    gs_cbc.fit(X_train, y_train)
    best_model = gs_cbc.best_estimator_

    pickle.dump(best_model, open(CBC_PATH, 'wb'))

    return best_model, roc_auc_score(y_val, best_model.predict_proba(X_val)[:, 1])


def rfc_tuning(RFC_PATH: str, X_train, X_val, y_train, y_val, cv):
    """
    Подбор гиперпараметров и обучение модели с лучшими гиперпараметрами

    :param RFC_PATH: путь для сохранения модели;
    :param X_train: выборка для обучения;
    :param X_val: выборка для валидации;
    :param y_train: таргет для обучения;
    :param y_val: таргет для валидации;
    :param cv: алгоритм кросс-валидации;
    :return: обученная модели, скор на выборке для валидации
    """
    rfc = RandomForestClassifier(random_state=22)

    params_rfc = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10]
    }
    gs_rfc = GridSearchCV(rfc, param_grid=params_rfc, cv=cv, scoring='roc_auc')
    gs_rfc.fit(X_train, y_train)
    best_model = gs_rfc.best_estimator_
    pickle.dump(best_model, open(RFC_PATH, 'wb'))

    return best_model, roc_auc_score(y_val, best_model.predict_proba(X_val)[:, 1])


def blending(logreg, dtc, cbc, rfc, X_val, y_val) -> float:
    """
    Считаем скоры моделей, на них обучаем мета-модель

    :param logreg: модель логистической регрессии;
    :param dtc: модель случайного леса;
    :param cbc: модель градиентного бустинга;
    :param rfc: модель случайного леса;
    :param X_val: выборка для валидации;
    :param y_val: таргет для валидации;
    :return: получившийся скор
    """
    preds_cbc = cbc.predict_proba(X_val)[:, 1]
    preds_rfc = rfc.predict_proba(X_val)[:, 1]
    preds_dtc = dtc.predict_proba(X_val)[:, 1]
    preds_logreg = logreg.predict_proba(X_val)[:, 1]

    all_preds = np.array([preds_cbc, preds_rfc, preds_dtc, preds_logreg])

    blender = LogisticRegression(random_state=22)
    blender.fit(all_preds.T, y_val)
    total_predict = blender.predict_proba(all_preds.T)

    return roc_auc_score(y_val, total_predict[:, 1])


def stacking(logreg, dtc, cbc, rfc, X_train, X_val, y_train, y_val, cv):
    """
    Организуем стэкинг на основе предсказаний 4 моделей, мета-моделью выступает логистическая регрессия

    :param logreg: модель логистической регрессии;
    :param dtc: модель случайного леса;
    :param cbc: модель градиентного бустинга;
    :param rfc: модель случайного леса;
    :param X_train: выборка для обучения;
    :param X_val: выборка для валидации;
    :param y_train: таргет для обучения;
    :param y_val: таргет для валидации;
    :param cv: алгоритм кросс-валидации;
    :return: скор стекинга
    """
    estimators = [('cbc', cbc),
                  ('dtc', dtc),
                  ('logreg', logreg),
                  ('rf_clf', rfc)]

    meta_model = LogisticRegression(random_state=22)
    stacking_model = StackingClassifier(estimators=estimators, final_estimator=meta_model, cv=cv)

    stacking_model.fit(X_train, y_train)

    return roc_auc_score(y_val, stacking_model.predict_proba(X_val)[:, 1])


def main(DATA_PATH: str, files_names: list, LOGREG_PATH: str, DTC_PATH: str,
         CBC_PATH: str, RFC_PATH: str) -> None:
    """
    Склеиваем данные, подбираем параметры для градиентного бустинга и случайного леса,
    используем блендинг и стекинг на обученных моделях, печатаем получившиеся скоры
    :param DATA_PATH: путь к таблицам с признаками;
    :param files_names: список с названиями таблиц;
    :param LOGREG_PATH: путь к модели логистической регрессии;
    :param DTC_PATH: путь к модели дерева решений;
    :param CBC_PATH: путь для сохранения модели градиентного бустинга;
    :param RFC_PATH: путь для сохранения модели случайного леса
    """
    df = create_dataframe(DATA_PATH, files_names)
    df_train = df.loc[df['target'].isin([0, 1])]

    X = df_train.drop(columns='target')
    y = df_train['target']

    X_train, X_val, y_train, y_val = train_test_split(X, y, shuffle=True, random_state=22, train_size=0.8)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=22)

    logreg = pickle.load(open(LOGREG_PATH, 'rb'))
    dtc = pickle.load(open(DTC_PATH, 'rb'))
    cbc, cbc_score = cbc_tuning(CBC_PATH, X_train, X_val, y_train, y_val, cv)
    rfc, rfc_score = rfc_tuning(RFC_PATH, X_train, X_val, y_train, y_val, cv)

    blending_score = blending(logreg, dtc, cbc, rfc, X_val, y_val)
    stacking_score = stacking(logreg, dtc, cbc, rfc, X_train, X_val, y_train, y_val, cv)

    print(f"ROC-AUC скор catboost: {cbc_score},\n ROC-AUC скор rfc: {rfc_score},"
          f"\n ROC-AUC скор blending: {blending_score},\n ROC-AUC скор stacking: {stacking_score}")


main(DATA_PATH, files_names, LOGREG_PATH, DTC_PATH, CBC_PATH, RFC_PATH)