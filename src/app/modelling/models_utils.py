import pickle
from features_utils import features_selection
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split


def tuning(DATA_PATH: str, files_names: list, OUTPUT_PATH: str,) -> float:
    """Подготовка данных для обучения, поиск гиперпараметров модели, подсчет скора и сохранение лучшей модели."""
    df_train, meaningful_numeric = features_selection(DATA_PATH, files_names)
    X = df_train.drop(columns='target')
    y = df_train['target']

    X_train, X_val, y_train, y_val = train_test_split(X, y, shuffle=True, random_state=22, train_size=0.8)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=22)

    if 'dtc' in OUTPUT_PATH:
        model = DecisionTreeClassifier(random_state=22)

        params = {'max_depth': [5, 10, 15],
                  'min_samples_split': [200, 300, 500]}

    else:
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), meaningful_numeric)
            ])
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', LogisticRegression(random_state=22, max_iter=2000))
        ])

        params = {'model__C': [0.001, 0.01, 0.1],
                  'model__solver': ['lbfgs', 'liblinear']}

    gs = GridSearchCV(model, param_grid=params, scoring='roc_auc', cv=cv, refit=True)
    gs.fit(X_train, y_train)
    best_model = gs.best_estimator_
    pickle.dump(best_model, open(OUTPUT_PATH, 'wb'))

    return roc_auc_score(y_val, best_model.predict_proba(X_val)[:, 1])