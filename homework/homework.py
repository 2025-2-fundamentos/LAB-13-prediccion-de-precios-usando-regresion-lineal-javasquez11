#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#
# flake8: noqa: E501
# flake8: noqa: E501
"""
Predicción de precios de vehículos usados con Regresión Lineal
"""
import os
import gzip
import pickle
import json
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def load_data(path: str) -> pd.DataFrame:
    """Carga datos desde un archivo CSV comprimido."""
    return pd.read_csv(path, index_col=False, compression='zip')


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesa el dataset."""
    df = df.copy()
    
    # Crear columna Age (2021 - Year)
    df['Age'] = 2021 - df['Year']
    
    # Eliminar columnas Year y Car_Name
    df = df.drop(columns=['Year', 'Car_Name'])
    
    return df


def create_pipeline() -> Pipeline:
    """Crea el pipeline de preprocesamiento y modelado."""
    
    # Índices de columnas (después de preprocesar):
    # 0=Present_Price, 1=Driven_kms, 2=Fuel_Type, 3=Selling_type, 
    # 4=Transmission, 5=Owner, 6=Age
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), [0, 1, 5, 6]),  # Numéricas
            ('cat', OneHotEncoder(handle_unknown='ignore'), [2, 3, 4]),  # Categóricas
        ]
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('k_best', SelectKBest(score_func=f_regression)),
        ('model', LinearRegression())
    ])
    
    return pipeline


def create_estimator(pipeline: Pipeline) -> GridSearchCV:
    """Crea el GridSearchCV para optimizar hiperparámetros."""
    
    param_grid = {
        'k_best__k': range(1, 11),
    }
    
    return GridSearchCV(
        pipeline,
        param_grid,
        cv=10,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        refit=True
    )


def save_model(model, path: str = 'files/models/model.pkl.gz'):
    """Guarda el modelo comprimido."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with gzip.open(path, 'wb') as f:
        pickle.dump(model, f)


def calculate_and_save_metrics(model, X_train, X_test, y_train, y_test):
    """Calcula y guarda las métricas del modelo."""
    
    # Predicciones
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Métricas de entrenamiento
    train_metrics = {
        'type': 'metrics',
        'dataset': 'train',
        'r2': float(r2_score(y_train, y_train_pred)),
        'mse': float(mean_squared_error(y_train, y_train_pred)),
        'mad': float(mean_absolute_error(y_train, y_train_pred)),
    }
    
    # Métricas de prueba
    test_metrics = {
        'type': 'metrics',
        'dataset': 'test',
        'r2': float(r2_score(y_test, y_test_pred)),
        'mse': float(mean_squared_error(y_test, y_test_pred)),
        'mad': float(mean_absolute_error(y_test, y_test_pred)),
    }
    
    # Guardar métricas
    os.makedirs('files/output', exist_ok=True)
    output_path = 'files/output/metrics.json'
    
    with open(output_path, 'w') as f:
        f.write(json.dumps(train_metrics) + '\n')
        f.write(json.dumps(test_metrics) + '\n')


def main():
    """Función principal que ejecuta el pipeline completo."""
    
    # Paso 1: Cargar y preprocesar datos
    train_df = preprocess_data(load_data('files/input/train_data.csv.zip'))
    test_df = preprocess_data(load_data('files/input/test_data.csv.zip'))
    
    # Paso 2: Dividir en X e y y convertir a numpy arrays
    x_train = train_df.drop(columns=['Selling_Price']).values
    y_train = train_df['Selling_Price'].values
    
    x_test = test_df.drop(columns=['Selling_Price']).values
    y_test = test_df['Selling_Price'].values
    
    # Paso 3: Crear pipeline
    pipeline = create_pipeline()
    
    # Paso 4: Optimizar hiperparámetros con GridSearchCV
    model = create_estimator(pipeline)
    model.fit(x_train, y_train)
    
    # Paso 5: Guardar modelo
    save_model(model)
    
    # Paso 6: Calcular y guardar métricas
    calculate_and_save_metrics(model, x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    main()






