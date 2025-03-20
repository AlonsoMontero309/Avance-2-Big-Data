import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout
import matplotlib.pyplot as plt

# Cargar los datos
data = pd.read_csv('data/datos_retail.csv') 

X = data[['encoded_category', 'quantity']].values  # Características de entrada
y = data['total_amount'].values  # Variable objetivo

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Definir el modelo
model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))  # Capa de entrada
model.add(Dense(64, activation='relu'))  # Capa oculta
model.add(Dropout(0.2))  # Capa de Dropout
model.add(Dense(32, activation='relu'))  # Capa oculta
model.add(Dense(1))  # Capa de salida

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

# Evaluar el modelo
train_loss = model.evaluate(X_train, y_train)
print(f'Training Loss: {train_loss}')

# Graficar la pérdida
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()
