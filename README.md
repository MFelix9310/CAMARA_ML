# Predicción de Precios de Cámaras Digitales: Proyecto de Machine Learning

## Descripción del Proyecto

Este proyecto tiene como objetivo desarrollar un modelo de Machine Learning de nivel senior para predecir el **precio (`Price`)** de cámaras digitales basándose en sus características técnicas y fecha de lanzamiento. Se utiliza un enfoque integral que abarca desde el análisis exploratorio de datos y la ingeniería de características hasta el entrenamiento, la optimización y la evaluación de múltiples modelos de regresión, culminando en una interfaz interactiva para predicciones.

El proyecto está diseñado para ser un ejemplo completo y robusto, adecuado para un portafolio de Data Scientist o Machine Learning Engineer.

## Dataset

*   **Nombre del Archivo**: `camera_dataset.csv`
*   **Fuente**: El dataset fue proporcionado por el usuario. Originalmente parece estar basado en el "1000 Cameras Dataset" que se puede encontrar en plataformas como Kaggle.
*   **Descripción Breve**: Contiene información sobre aproximadamente 1000 modelos de cámaras digitales, incluyendo especificaciones como resolución, zoom, píxeles efectivos, fecha de lanzamiento y precio.
*   **Ubicación**: El dataset debe estar ubicado en la carpeta `data/` dentro de la estructura del proyecto.

## Estructura del Proyecto

```
camera_price_prediction_project/
├── data/
│   └── camera_dataset.csv
├── notebooks/
│   └── camera_price_prediction.ipynb
├── src/  (Opcional: para scripts auxiliares o funciones)
├── images/ (Opcional: para guardar gráficos generados si se referencian en el README)
├── requirements.txt (Recomendado: generar con `pip freeze > requirements.txt`)
└── README.md
```

## Instalación y Dependencias

Para ejecutar este proyecto, necesitarás Python 3.x y las siguientes librerías principales. Se recomienda crear un entorno virtual.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm ipywidgets nbformat jupyter
```

Alternativamente, si se proporciona un archivo `requirements.txt` en el futuro, puedes instalar todas las dependencias con:

```bash
pip install -r requirements.txt
```

## Uso

1.  Clona o descarga este repositorio.
2.  Asegúrate de que el archivo `camera_dataset.csv` esté en la carpeta `data/`.
3.  Navega a la carpeta `notebooks/`.
4.  Abre y ejecuta el Jupyter Notebook `camera_price_prediction.ipynb` en un entorno Jupyter (Jupyter Lab o Jupyter Notebook).

    ```bash
    cd camera_price_prediction_project/notebooks
    jupyter lab camera_price_prediction.ipynb
    # o
    # jupyter notebook camera_price_prediction.ipynb
    ```
5.  Sigue las celdas del notebook para ver el análisis, el entrenamiento del modelo y la interfaz de predicción interactiva al final.

## Resumen del Pipeline de Machine Learning

1.  **Carga y Comprensión de Datos**: Se carga el dataset y se realiza una inspección inicial.
2.  **Análisis Exploratorio de Datos (EDA)**: Se analiza la variable objetivo (`Price`) y otras características, se visualizan distribuciones, relaciones y correlaciones.
3.  **Preprocesamiento e Ingeniería de Características**: 
    *   Manejo de valores nulos.
    *   Creación de nuevas características (ej. `Camera Age`, `Zoom Ratio`, `Resolution Area`).
    *   Escalado de características numéricas.
4.  **División de Datos**: Separación en conjuntos de entrenamiento y prueba.
5.  **Entrenamiento de Modelos**: Se entrenan y evalúan (con validación cruzada) múltiples modelos de regresión (Lineales, KNN, Árboles de Decisión, Random Forest, Gradient Boosting, XGBoost, LightGBM).
6.  **Optimización de Hiperparámetros**: Se utiliza `RandomizedSearchCV` (o similar) para optimizar los hiperparámetros del modelo más prometedor.
7.  **Evaluación Final e Interpretación**: El modelo final se evalúa en el conjunto de prueba. Se analizan métricas como RMSE, MAE, R² y se visualiza la importancia de las características.
8.  **Interfaz de Predicción Interactiva**: Se implementa una interfaz con `ipywidgets` para permitir al usuario ingresar características y obtener una predicción de precio.

## Resultados Clave (Ejemplo - Completar después de la ejecución)

*   **Mejor Modelo Seleccionado**: [Ej: LightGBM Regressor]
*   **Métricas en el Conjunto de Prueba (Escala Original del Precio)**:
    *   RMSE: $[Valor RMSE]
    *   MAE: $[Valor MAE]
    *   R²: [Valor R²]
    *   MAPE: [Valor MAPE]%
*   **Características Más Influyentes**: [Ej: `Effective pixels`, `Camera Age`, `Max resolution`]

## Limitaciones y Posibles Mejoras

*   **Limitaciones**: 
    *   El tamaño del dataset es moderado.
    *   La información de "Marca" no se extrajo explícitamente del nombre del modelo, lo que podría ser una feature útil.
    *   Algunas características como `Dimensions` son ambiguas.
*   **Posibles Mejoras**: 
    *   Incorporar datos más recientes o de mayor volumen.
    *   Extraer la marca de la cámara como una característica categórica.
    *   Probar técnicas de encoding más avanzadas para características categóricas (si se añaden).
    *   Explorar modelos de Deep Learning si el dataset crece.
    *   Desplegar el modelo como una API web.

---

*Este README fue generado como parte de un proyecto de demostración.*

