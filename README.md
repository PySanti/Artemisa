
# Project Title

A brief description of what this project does and who it's for

# Artemisa

El objetivo de este proyecto será ejecutar un ejercicio de aprendizaje supervisado de regresión implementando los algoritmos: *KNN, Ridge, Lasso y Regresión Lineal*.

Ejecutaremos los pasos formales en el proceso de preprocesamiento y entrenamiento a través de la *matriz de estudio* vista en otros ejercicios.

La idea del proyecto será revisar el rendimiento de *KNN* (un algoritmo que aprendimos hace poco), y *comparar los resultados de Ridge y Lasso con Regresión Lineal*.

El [dataset](https://www.kaggle.com/datasets/ebrahimhaquebhatti/pakistan-house-price-prediction) a utilizar contiene informacion acerca de casas en pakistan. La idea es crear un modelo capaz de predecir el precio de una casa dada sus características.


## Preprocesamiento

Primeramente se eliminaron las columnas `Unnamed: 0`, `page_url`, `property_id`.

Shape del dataset : `168446 x 18`

### Duplicados

Se encontraron `15011` duplicados.

### Manejo de Nans

Las unicas columnas que contienen valores Nan son:

```
agency  26.16%
agent   26.16%
```

Los registros que son null en agency tambien son null en agent. Se eliminaran usando el siguiente comando:

```
df1 = df1.drop(df1.query("agency != agency and agent != agent").index)
df1 = df1.drop(155597)
```

### Codificación

Las columnas con valores categoricos son:


```
property_type:  one-hot-encoding
city:           one-hot-encoding
province_name:  one-hot-encoding
purpose:        one-hot-encoding
location:       target-encoding
agency:         target-encoding
agent:          target-encoding
date-added:     date-encoder
```

### Scalers

El scaling sera aplicado sobre las siguientes columnas:

```
location_id
price
latitude
longitude
baths
bedrooms
Total_Area
```

### Extracción y/o selección de características

### Outliers

### Campana de Gauss

Al ser un problema de regresión, no se revisará.

### Estudio de correlaciones

No se revisará.

### Desequilibrio de target

De nuevo, al ser un problema de regresión, no se revisará.


## Entrenamiento

## Evaluación%      