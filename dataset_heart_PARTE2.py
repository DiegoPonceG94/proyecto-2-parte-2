# Cargar librerías
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import optuna
from sklearn.model_selection import cross_val_score

# Cargar dataset
from google.colab import drive
drive.mount('/content/drive')

# Importamos el DataFrame.
path = "/content/drive/MyDrive/datasets/dataset_heart.csv"
df = pd.read_csv(path)

# Vista general
print("Primeras filas:")
display(df.head())

print("\nResumen de tipos de datos:")
display(df.info())

# --- EDA ---

# 1. VALORES NULOS
print("\nConteo de valores nulos por columna:")
print(df.isnull().sum())

# Si hay pocos valores nulos, se pueden eliminar las filas
# Si hay más, imputar según el tipo de dato (media, mediana, moda)

# Ejemplo general de imputación:
# Por tipo de dato
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype in ['float64', 'int64']:
            median = df[col].median()
            df[col].fillna(median, inplace=True)
        else:
            mode = df[col].mode()[0]
            df[col].fillna(mode, inplace=True)

print("\nValores nulos después de la imputación:")
print(df.isnull().sum())

# 2. OUTLIERS
# Usamos boxplots para identificar outliers en variables numéricas

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

for col in numeric_cols:
    plt.figure(figsize=(6, 3))
    sns.boxplot(data=df, x=col, color='skyblue')
    plt.title(f'Distribución y outliers: {col}')
    plt.tight_layout()
    plt.show()

# Eliminación de outliers usando el método IQR

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower) & (df[column] <= upper)]
    return filtered_df

# Aplicamos IQR para columnas con outliers claros (ejemplo: 'chol', 'trestbps')
cols_to_filter = ['chol', 'trestbps', 'thalach', 'oldpeak']  # personalizable

for col in cols_to_filter:
    original_shape = df.shape
    df = remove_outliers_iqr(df, col)
    new_shape = df.shape
    print(f"Filtrados outliers en '{col}': {original_shape[0] - new_shape[0]} filas eliminadas.")

# --- TRANSFORMACION COLUMNAS ---

# Separar variables numéricas y categóricas
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

# También incluimos como categóricas algunas variables tipo int que son en realidad cualitativas
# como 'cp', 'thal', 'slope', etc., si están en el dataset como int

# Supongamos que estas son categóricas codificadas como int:
cols_categoricas_extra = ['cp', 'thal', 'slope', 'restecg', 'ca', 'sex', 'fbs', 'exang']
for col in cols_categoricas_extra:
    if col in df.columns and col not in cat_cols:
        cat_cols.append(col)
        if col in num_cols:
            num_cols.remove(col)

# Separar X (features) e y (target)
X = df.drop(columns=['target']) if 'target' in df.columns else df.drop(columns=['Target'])
y = df['target'] if 'target' in df.columns else df['Target']

# Crear el ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ]
)

# Aplicamos la transformación
X_transformed = preprocessor.fit_transform(X)

# Mostrar resultado
print("Forma de X antes:", X.shape)
print("Forma de X después de la transformación:", X_transformed.shape)


# --- PIPELINE ---

# --- Definir columnas numéricas y categóricas  ---
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

cols_categoricas_extra = ['cp', 'thal', 'slope', 'restecg', 'ca', 'sex', 'fbs', 'exang']
for col in cols_categoricas_extra:
    if col in df.columns and col not in cat_cols:
        cat_cols.append(col)
        if col in num_cols:
            num_cols.remove(col)

# Separar X y y
X = df.drop(columns=['target']) if 'target' in df.columns else df.drop(columns=['Target'])
y = df['target'] if 'target' in df.columns else df['Target']

# --- Pipelines para imputación y preprocesamiento ---
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # imputa con mediana
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
])

# Pipeline final con modelo
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar pipeline
model_pipeline.fit(X_train, y_train)

# Predecir y evaluar
y_pred = model_pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# --- ENTRENAMIENTO INICIAL ---

# Lista de modelos a evaluar
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'LightGBM': lgb.LGBMClassifier(random_state=42)
}

# Diccionario para guardar resultados
results = {}

for name, model in models.items():
    print(f"Evaluando: {name}")
    
    # Crear pipeline con preprocesador + modelo
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Validación cruzada con 5 folds, métrica accuracy (puedes usar scoring='f1' u otra)
    scores = cross_val_score(pipe, X, y, cv=5, scoring='accuracy')
    
    results[name] = scores
    print(f"Accuracy promedio: {scores.mean():.4f} ± {scores.std():.4f}\n")

# Mostrar resumen de resultados
print("Resumen de modelos evaluados:")
for name, scores in results.items():
    print(f"{name}: Accuracy media = {scores.mean():.4f}, std = {scores.std():.4f}")

# Mejor modelo
best_model_name = max(results, key=lambda k: results[k].mean())
print(f"\nMejor modelo inicial según accuracy: {best_model_name}")

# --- OPTIMIZACION DE HIPERPARAMETROS --- 

# Espacio de búsqueda (hiperparámetros relevantes para Random Forest)
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

# Crear pipeline con preprocesador y RF base
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# GridSearchCV con 5 folds
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

# Ejecutar búsqueda
grid_search.fit(X_train, y_train)

print("Mejores hiperparámetros encontrados con GridSearchCV:")
print(grid_search.best_params_)
print(f"Mejor score: {grid_search.best_score_:.4f}")

# Espacio de búsqueda más amplio (distribuciones)
param_dist = {
    'classifier__n_estimators': randint(50, 300),
    'classifier__max_depth': [None] + list(range(5, 50, 5)),
    'classifier__min_samples_split': randint(2, 20),
    'classifier__min_samples_leaf': randint(1, 20),
    'classifier__bootstrap': [True, False]
}

random_search = RandomizedSearchCV(
    pipeline, param_distributions=param_dist, 
    n_iter=50, cv=5, scoring='accuracy', n_jobs=-1, verbose=1, random_state=42
)

random_search.fit(X_train, y_train)

print("Mejores hiperparámetros encontrados con RandomizedSearchCV:")
print(random_search.best_params_)
print(f"Mejor score: {random_search.best_score_:.4f}")

def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 5, 50)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
    bootstrap = trial.suggest_categorical('bootstrap', [True, False])
    
    # Crear modelo con parámetros sugeridos
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        bootstrap=bootstrap,
        random_state=42
    )
    
    # Pipeline completo
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    
    # Validación cruzada 5 folds
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
    
    return scores.mean()

# Crear estudio y optimizar
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print("Mejores hiperparámetros encontrados con Optuna:")
print(study.best_params)
print(f"Mejor score: {study.best_value:.4f}")

# Usando mejores parámetros de Optuna, GridSearchCV o RandomizedSearchCV
best_params = study.best_params  # o grid_search.best_params_, random_search.best_params_

# Crear modelo con mejores hiperparámetros
best_rf = RandomForestClassifier(
    n_estimators=best_params.get('n_estimators', 100),
    max_depth=best_params.get('max_depth', None),
    min_samples_split=best_params.get('min_samples_split', 2),
    min_samples_leaf=best_params.get('min_samples_leaf', 1),
    bootstrap=best_params.get('bootstrap', True),
    random_state=42
)

# Pipeline final
final_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', best_rf)
])

# Entrenar con todo el set de entrenamiento
final_pipeline.fit(X_train, y_train)

# Predecir en test
y_pred = final_pipeline.predict(X_test)

# Evaluar
print("Reporte clasificación en conjunto de prueba con modelo optimizado:")
print(classification_report(y_test, y_pred))

# Modelo inicial sin hiperparámetros ajustados
initial_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])
initial_pipeline.fit(X_train, y_train)
y_pred_init = initial_pipeline.predict(X_test)

print("Reporte clasificación en conjunto de prueba con modelo inicial:")
print(classification_report(y_test, y_pred_init))