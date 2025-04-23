from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
import pandas as pd
import seaborn as snus
import matplotlib.pyplot as plt

df = pd.read_csv('mental_health/data/train.csv')

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df.drop(columns=['Name', 'id'], inplace=True)
    df['CGPA'].fillna(0, inplace=True)

    df['Pressure'] = df['Academic Pressure'].fillna(0) + df['Work Pressure'].fillna(0)
    df = df.drop(columns=['Academic Pressure', 'Work Pressure'])
    df['Satisfaction'] = df['Study Satisfaction'].fillna(0) + df['Job Satisfaction'].fillna(0)
    df = df.drop(columns=['Study Satisfaction', 'Job Satisfaction'])

    df['Profession'] = df[df['Working Professional or Student'] == 'Working Professional']['Profession'].apply(lambda x: x if pd.notna(x) else 'Unknown')
    df['Profession'] = df['Profession'].fillna('Student')

    df = pd.get_dummies(df, dtype=int, prefix=['', '', 'Familly illnes: ', 'Suicadal thoughts: '], prefix_sep="", columns=['Working Professional or Student', 'Gender', 'Family History of Mental Illness', 'Have you ever had suicidal thoughts ?'], drop_first=False)
    city_encoder = LabelEncoder()
    profession_encoder = LabelEncoder()
    degree_encoder = LabelEncoder()
    df['City'] = city_encoder.fit_transform(df['City'])
    df['Sleep Duration'] = df['Sleep Duration'].apply(lambda x: 0 if x == 'Less than 5 hours' else (1 if x == '5-6 hours' else (2 if x == '5-7 hours' else (3 if x == '7-8 hours' else 4))))
    df['Profession'] = profession_encoder.fit_transform(df['Profession'])
    df['Dietary Habits'] = df['Dietary Habits'].apply(lambda x: 0 if x == 'Unhealthy' else (1 if x == 'Moderate' else 2))
    df['Degree'] = degree_encoder.fit_transform(df['Degree'])
    return df

df = preprocess_data(df)

def grid_search(model, param_grid, X_train, y_train):
    pipeline = Pipeline([
        ('feature_selection', RFECV(estimator=LogisticRegression(random_state=42), step=1, cv=StratifiedKFold(5), scoring='f1')),
        ('classifier', model)
    ])

    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=StratifiedKFold(5), scoring='f1', n_jobs=-1)

    grid_search.fit(X_train, y_train)

    print("Best parameters found: ", grid_search.best_params_)
    print("Best cross-validation score: ", grid_search.best_score_)

model = LogisticRegression(random_state=42)
param_grid = {
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__penalty': ['l1', 'l2', 'elasticnet'],
    'classifier__solver': ['liblinear', 'saga', 'lbfgs'],
    'classifier__max_iter': [100, 500, 1000],
    'classifier__class_weight': [None, 'balanced']
}

df_train, df_test = train_test_split(df, test_size=0.1, random_state=42)
X_train = df_train.drop(columns=['Depression'])
y_train = df_train['Depression']
X_test = df_test.drop(columns=['Depression'])
y_test = df_test['Depression']
# grid_search(model, param_grid, X_train, y_train) 

# model = LogisticRegression(random_state=42)
# Best parameters found:  {'classifier__C': 10, 'classifier__class_weight': 'balanced', 'classifier__max_iter': 100, 'classifier__penalty': 'l1', 'classifier__solver': 'liblinear'}
# Best cross-validation score:  0.9660556448843739

model = RandomForestClassifier(random_state=42, n_jobs=-1)
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__max_features': ['auto', 'sqrt', 'log2'],
    'classifier__class_weight': [None, 'balanced']
}

df_train, df_test = train_test_split(df, test_size=0.1, random_state=42)
X_train = df_train.drop(columns=['Depression'])
y_train = df_train['Depression']
X_test = df_test.drop(columns=['Depression'])
y_test = df_test['Depression']
# grid_search(model, param_grid, X_train, y_train) 

# model = RandomForestClassifier(random_state=42, n_jobs=-1)
# Best parameters found:  {'classifier__class_weight': 'balanced', 'classifier__max_depth': 10, 'classifier__max_features': 'sqrt', 'classifier__min_samples_leaf': 2, 'classifier__min_samples_split': 5, 'classifier__n_estimators': 200}
# Best cross-validation score:  0.8354104649870525

def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=100, random_state=42, C=10, penalty='l1', solver='liblinear', class_weight='balanced')
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print("F1 Score: ", f1)

# model = train_model(X_train, y_train)
# evaluate_model(model, X_test, y_test)

df_to_predict = pd.read_csv('mental_health/data/test.csv')
df_to_predict = preprocess_data(df_to_predict)

model = train_model(df.drop(columns=['Depression']), df['Depression'])
y_predicted = model.predict(df_to_predict)

df_predicted = pd.DataFrame(y_predicted, columns=['Depression'])
df_predicted['id'] = pd.read_csv('mental_health/data/test.csv')['id']
df_predicted = df_predicted[['id', 'Depression']]

df_predicted.to_csv('mental_health/data/predictions.csv', index=False)




