# README del modelo de predicción de deserción estudiantil

## Descripción
Este proyecto genera un conjunto de datos sintéticos que simula el comportamiento académico y socioeconómico de estudiantes de Ingeniería de Sistemas tras la implementación del nuevo pensum. A partir de estos datos se entrena una red neuronal para predecir la probabilidad de deserción estudiantil.

## Flujo del proceso
1. **Generación de datos simulados**  
   Se crean 5000 registros ficticios con variables como semestre actual, número de materias difíciles, promedio académico, materias reprobadas, estrato socioeconómico, si trabaja y si tiene beca. También se calcula una probabilidad de deserción basada en reglas realistas.

2. **Preparación de datos**  
   - Se separan las variables predictoras (X) y la variable objetivo (y: deserta o no).
   - Se dividen los datos en entrenamiento, validación y prueba.
   - Se escalan las variables numéricas para optimizar el entrenamiento.

3. **Definición del modelo**  
   - Red neuronal multicapa con:
     - Capa de entrada con tantas neuronas como variables.
     - Capas ocultas con activación ReLU y dropout para evitar sobreajuste.
     - Capa de salida con activación sigmoide para clasificación binaria.

4. **Entrenamiento**  
   - Se utiliza optimizador Adam, pérdida binaria y métricas de exactitud y AUC.
   - Se aplica early stopping para detener el entrenamiento si no hay mejoras.

5. **Evaluación**  
   - Se evalúa en el conjunto de prueba mostrando:
     - Pérdida (loss)
     - Exactitud (accuracy)
     - AUC (área bajo la curva ROC)
   - Se generan métricas como matriz de confusión, reporte de clasificación y curva ROC.
   - Se visualizan las curvas de entrenamiento y validación.

6. **Guardado del modelo**  
   - El modelo entrenado se guarda como `modelo_desercion.h5` para su uso posterior.

## Objetivo
Predecir el riesgo de deserción de un estudiante en función de sus características académicas y socioeconómicas, permitiendo identificar casos de alto riesgo y proyectar el impacto del nuevo pensum académico.

## Tecnologías utilizadas
- Python 3
- NumPy y Pandas para manejo de datos
- Scikit-learn para preprocesamiento y métricas
- TensorFlow/Keras para la red neuronal
- Matplotlib para visualización

## Salida esperada
- Métricas de desempeño del modelo (pérdida, exactitud y AUC)
- Matriz de confusión y reporte de clasificación
- Gráficas de pérdida y exactitud por época
- Archivo `modelo_desercion.h5` con el modelo entrenado listo para ser cargado y utilizado.
