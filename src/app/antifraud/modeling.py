import pandas as pd
from rules import filter_clients
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold


DATA_PATH = "mfeatures.csv"


def cbc_tuning(X_train, X_val, y_train, y_val, cv):
    """Подбор гиперпараметров модели, подсчет скора для лучшей модели."""
    cbc = CatBoostClassifier(random_state=22, verbose=False, eval_metric='AUC')

    params_cbc = {
        'learning_rate': [0.01, 0.001],
        'iterations': [100, 200],
        'depth': [5, 10],
    }
    gs_cbc = GridSearchCV(cbc, param_grid=params_cbc, cv=cv, scoring='roc_auc')
    gs_cbc.fit(X_train, y_train)
    best_model = gs_cbc.best_estimator_

    return roc_auc_score(y_val, best_model.predict_proba(X_val)[:, 1])


def main(DATA_PATH: str):
    """Используем правила антифрода, обучаем и валидируем модель на очищенной выборке."""
    df = pd.read_csv(DATA_PATH)
    df_clean = filter_clients(df)

    df_train = df_clean.loc[df_clean['target'].isin([0, 1])]
    X = df_train.drop(columns='target')
    y = df_train['target']

    X_train, X_val, y_train, y_val = train_test_split(X, y, shuffle=True, random_state=22, train_size=0.8)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=22)

    print(f"ROC_AUC скор catboost: {cbc_tuning(X_train, X_val, y_train, y_val, cv)}")  # 0.7324218475910278


main(DATA_PATH)

