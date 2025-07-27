import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

np.random.seed(42)

num_registros = 5000

semestres = np.random.randint(1, 9, num_registros)
creditos_semestre = np.full(num_registros, 18)

materias_dificiles = []
for s in semestres:
    if s in [1, 2]:
        materias_dificiles.append(np.random.randint(2, 4))
    elif s in [3, 4]:
        materias_dificiles.append(np.random.randint(3, 5))
    elif s in [5, 6]:
        materias_dificiles.append(np.random.randint(2, 3))
    else:
        materias_dificiles.append(np.random.randint(1, 3))

materias_dificiles = np.array(materias_dificiles)

promedio_academico = np.clip(np.random.normal(3.5, 0.5, num_registros), 2.5, 5.0)
materias_reprobadas = np.random.poisson(1, num_registros)
estrato_socioeconomico = np.random.randint(1, 7, num_registros)
trabaja_mientras_estudia = np.random.randint(0, 2, num_registros)
tiene_beca = np.random.randint(0, 2, num_registros)

prob_base = 0.15
prob_desercion = (
    prob_base
    + (materias_dificiles * 0.05)
    + ((3.0 - promedio_academico) * 0.1)
    + (trabaja_mientras_estudia * 0.1)
    - (tiene_beca * 0.05)
    + (1 - estrato_socioeconomico / 6) * 0.1
)

prob_desercion = np.clip(prob_desercion, 0, 1)
deserta = np.random.binomial(1, prob_desercion)

datos = pd.DataFrame({
    'semestre_actual': semestres,
    'creditos_semestre': creditos_semestre,
    'materias_dificiles': materias_dificiles,
    'promedio_academico': promedio_academico,
    'materias_reprobadas': materias_reprobadas,
    'estrato_socioeconomico': estrato_socioeconomico,
    'trabaja_mientras_estudia': trabaja_mientras_estudia,
    'tiene_beca': tiene_beca,
    'deserta': deserta
})

X = datos.drop('deserta', axis=1).values
y = datos['deserta'].values

X_entrenamiento, X_temp, y_entrenamiento, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_validacion, X_prueba, y_validacion, y_prueba = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

escalador = StandardScaler()
X_entrenamiento = escalador.fit_transform(X_entrenamiento)
X_validacion = escalador.transform(X_validacion)
X_prueba = escalador.transform(X_prueba)

modelo = keras.Sequential([
    layers.Input(shape=(X_entrenamiento.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

modelo.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)

temprano_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

historial = modelo.fit(
    X_entrenamiento, y_entrenamiento,
    validation_data=(X_validacion, y_validacion),
    epochs=150,
    batch_size=32,
    callbacks=[temprano_stop],
    verbose=1
)

perdida, exactitud, auc = modelo.evaluate(X_prueba, y_prueba, verbose=0)
print(f"\nResultados en Prueba:\nPérdida: {perdida:.4f}\nExactitud: {exactitud:.4f}\nAUC: {auc:.4f}")

probabilidades = modelo.predict(X_prueba)
predicciones = (probabilidades > 0.5).astype(int)

print("\nMatriz de confusión:")
print(confusion_matrix(y_prueba, predicciones))

print("\nReporte de clasificación:")
print(classification_report(y_prueba, predicciones))

roc = roc_auc_score(y_prueba, probabilidades)
print(f"AUC Score: {roc:.4f}")

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(historial.history['loss'], label='Pérdida Entrenamiento')
plt.plot(historial.history['val_loss'], label='Pérdida Validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

plt.subplot(1,2,2)
plt.plot(historial.history['accuracy'], label='Exactitud Entrenamiento')
plt.plot(historial.history['val_accuracy'], label='Exactitud Validación')
plt.xlabel('Épocas')
plt.ylabel('Exactitud')
plt.legend()

plt.tight_layout()
plt.show()

modelo.save('modelo_desercion.h5')
