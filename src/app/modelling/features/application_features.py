import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(1, os.path.join(sys.path[0], '../../utils/'))
sys.path.insert(1, os.path.join(sys.path[0], '../../../config/'))

from db_connection import DB
from config import db_config


def weighted_average(col1, col2, col3):
    len_df = application_train_test.shape[0]
    w_1 = 1 - (col1.isna().sum()/len_df)
    w_2 = 1 - (col2.isna().sum()/len_df)
    w_3 = 1 - (col3.isna().sum()/len_df)
    return ((col1.fillna(0) * w_1 + col2.fillna(0) * w_2 + col3.fillna(0) * w_3)/(w_1 + w_2 + w_3)).replace(0, np.NaN)


def get_diff_train(row):
    return grouped_train.loc[row['code_gender']][row['name_education_type']] - row['amt_income_total']


def get_diff_test(row):
    return grouped_test.loc[row['code_gender']][row['name_education_type']] - row['amt_income_total']


def interest_rate_calc(row):
    return (row['amt_credit'] / row['amt_goods_price']) ** (1 / row['years']) - 1


db = DB(db_config)

application_train_test = db.get_df_from_query("""SELECT * FROM application_train_test""")
features = pd.DataFrame(application_train_test['sk_id_curr'])
# Кол-во документов
features['count_doc'] = application_train_test.loc[:, 'flag_document_2':'flag_document_21'].sum(axis=1)

# Есть ли полная информация о доме
features['flag_full_info_house'] = application_train_test.loc[:, 'apartments_avg':'emergencystate_mode']\
    .count(axis=1).apply(lambda x: 1 if x < 30 else 0)

# Кол-во полных лет
features['age'] = application_train_test['days_birth'].floordiv(-365)

# Год смены документа
features['year_change_doc'] = application_train_test['days_id_publish'].floordiv(-365)

# Разница во времени между сменой документа и возрастом на момент смены документы
features['diff_age_change_doc'] = features['age'] - features['year_change_doc']

# Признак задержки смены документа. Документ выдается или меняется в 14, 20 и 45 лет
features['flag_delay_change_doc'] = features['diff_age_change_doc']\
    .apply(lambda x: 0 if x == 14 or x == 20 or x == 45 else 1)

# Доля денег которые клиент отдает на займ за год
features['fraq_credit_income'] = application_train_test['amt_annuity'] / application_train_test['amt_income_total']

# Среднее кол-во детей в семье на одного взрослого
features['child_per_adult'] = application_train_test['cnt_children'] / \
                              (application_train_test['cnt_fam_members'] - application_train_test['cnt_children'])

# Средний доход на ребенка
features['income_per_child'] = (application_train_test['amt_income_total'] /
                                application_train_test['cnt_children']).apply(lambda x: 0 if x == float('inf') else x)

# Средний доход на взрослого
features['income_per_adult'] = (application_train_test['amt_income_total'] /
                                (application_train_test['cnt_fam_members'] - application_train_test['cnt_children']))

# Процентная ставка
application_train_test['years'] = (application_train_test['amt_credit'] / application_train_test['amt_annuity'])\
    .apply(lambda x: x if pd.isna(x) else round(x))
features['interest_rate'] = application_train_test[['amt_credit', 'years', 'amt_goods_price']]\
    .apply(interest_rate_calc, axis=1)

application_train_test['years'].drop(columns='years', inplace=True)

# Взвешенный скор внешних источников. Подумайте какие веса им задать.
features['weighted_ext_scores'] = weighted_average(application_train_test['ext_source_1'],
                                                   application_train_test['ext_source_2'],
                                                   application_train_test['ext_source_3'])

#  Разница между средним доходом в группе и доходом заявителя
app_train = application_train_test.loc[application_train_test['target'].isin([1, 0])]
app_test = application_train_test.loc[~(application_train_test['target'].isin([1, 0]))]

grouped_train = app_train.groupby(['code_gender', 'name_education_type'])['amt_income_total'].mean()
grouped_test = app_test.groupby(['code_gender', 'name_education_type'])['amt_income_total'].mean()

df_diff_train = app_train[['code_gender', 'name_education_type', 'amt_income_total']].apply(get_diff_train, axis=1)
df_diff_test = app_test[['code_gender', 'name_education_type', 'amt_income_total']].apply(get_diff_test, axis=1)

app_train['diff_amt_client_amt_group'] = app_train[['code_gender', 'name_education_type', 'amt_income_total']]\
    .apply(get_diff_train, axis=1)
app_test['diff_amt_client_amt_group'] = app_test[['code_gender', 'name_education_type', 'amt_income_total']]\
    .apply(get_diff_test, axis=1)

df_diff = pd.concat([app_train[['sk_id_curr', 'diff_amt_client_amt_group']],
                     app_test[['sk_id_curr', 'diff_amt_client_amt_group']]])

features = features.merge(df_diff, on='sk_id_curr')

features.set_index('sk_id_curr', inplace=True)
features.to_csv('features_csv/application_features.csv')
