
# Project Title

A brief description of what this project does and who it's for

# Artemisa

El objetivo de este proyecto será ejecutar un ejercicio de aprendizaje supervisado de regresión implementando los algoritmos: *KNN, Ridge, Lasso y Regresión Lineal*.

Ejecutaremos los pasos formales en el proceso de preprocesamiento.

La idea del proyecto será revisar el rendimiento de *KNN* (un algoritmo que aprendimos hace poco), y *comparar los resultados de Ridge y Lasso con Regresión Lineal*.

El [dataset](https://www.kaggle.com/datasets/ebrahimhaquebhatti/pakistan-house-price-prediction) a utilizar contiene informacion acerca de casas en pakistan. La idea es crear un modelo capaz de predecir el precio de una casa dada sus características.


## Preprocesamiento

Primeramente se eliminaron las columnas `Unnamed: 0`, `page_url`, `property_id`.

Shape del dataset : `168446 x 18`

### Duplicados

Se encontraron `15011 (8.91%)` duplicados.

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

Solo se hara extraccion de características.

### Outliers

No se estudiara

### Campana de Gauss

Al ser un problema de regresión, no se revisará.

### Estudio de correlaciones

No se revisará.

### Desequilibrio de target

De nuevo, al ser un problema de regresión, no se revisará.


## Entrenamiento

Se realizo un proceso iterativo de :

1- Obtencion de hiperparametros optimos para cada algoritmo

2- Evaluacion del algoritmo para esos hiperparametros y esa version de preprocesamiento.

Se evaluaron dos versiones de preprocesamiento : con y sin seleccion de características.

## Evaluación


### Variante con seleccion de características

```
Rendimiento de knn para train : 458143.832
Rendimiento de knn para test : 7771098.072
r2 de knn para train : 0.996
r2 de knn para test : 0.471
{'metric': 'manhattan', 'n_neighbors': 5, 'weights': 'distance'}
__________________________
Rendimiento de Ridge para train : 14303123.578
Rendimiento de Ridge para test : 14555746.583
r2 de Ridge para train : 0.392
r2 de Ridge para test : 0.332
{'alpha': 1e-06, 'fit_intercept': True, 'solver': 'svd', 'tol': 1e-07}
__________________________
/home/santiago/Escritorio/Aprendizaje ML/practicas/session8/Artemisa/dep/lib/python3.12/site-packages/sklearn/linear_model/_coordinate_descent.py:695: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 4.113e+19, tolerance: 1.354e+13
  model = cd_fast.enet_coordinate_descent(
Rendimiento de Lasso para train : 14303123.577
Rendimiento de Lasso para test : 14555746.583
r2 de Lasso para train : 0.392
r2 de Lasso para test : 0.332
{'alpha': 1e-06, 'fit_intercept': True, 'tol': 1e-07}
__________________________
Rendimiento de Regresion lineal para train : 14303123.578
Rendimiento de Regresion lineal para test : 14555746.583
r2 de Regresion lineal para train : 0.392
r2 de Regresion lineal para test : 0.332
{'copy_X': True, 'fit_intercept': True}
__________________________

```

### Variante sin seleccion de características

```
Rendimiento de knn para train : 9430158.551
Rendimiento de knn para test : 11664809.731
r2 de knn para train : 0.588
r2 de knn para test : 0.352
{'metric': 'manhattan', 'n_neighbors': 7, 'weights': 'uniform'}
__________________________
/home/santiago/Escritorio/Aprendizaje ML/practicas/session8/Artemisa/dep/lib/python3.12/site-packages/sklearn/linear_model/_ridge.py:215: LinAlgWarning: Ill-conditioned matrix (rcond=2.45444e-26): result may not be accurate.
  return linalg.solve(A, Xy, assume_a="pos", overwrite_a=True).T
Rendimiento de Ridge para train : 14103587.177
Rendimiento de Ridge para test : 14326474.730
r2 de Ridge para train : 0.416
r2 de Ridge para test : 0.360
{'alpha': 0.01, 'fit_intercept': False, 'solver': 'cholesky', 'tol': 1e-07}
__________________________
/home/santiago/Escritorio/Aprendizaje ML/practicas/session8/Artemisa/dep/lib/python3.12/site-packages/sklearn/linear_model/_coordinate_descent.py:695: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 3.954e+19, tolerance: 1.354e+13
  model = cd_fast.enet_coordinate_descent(
Rendimiento de Lasso para train : 14059392.304
Rendimiento de Lasso para test : 14288725.762
r2 de Lasso para train : 0.416
r2 de Lasso para test : 0.359
{'alpha': 1e-06, 'fit_intercept': True, 'tol': 1e-07}
__________________________
Rendimiento de Regresion lineal para train : 14022922.258
Rendimiento de Regresion lineal para test : 14260723.355
r2 de Regresion lineal para train : 0.414
r2 de Regresion lineal para test : 0.357
{'copy_X': True, 'fit_intercept': False}
__________________________
```