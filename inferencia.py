import pickle
import pandas as pd

# 1. Cargar los objetos guardados
with open('modelo_y_tratamientos.pkl', 'rb') as f:
    objetos = pickle.load(f)

modelo = objetos['modelo']
selector_varianza = objetos['selector_varianza']
discretizador = objetos['discretizador']

# 2. Cargar el conjunto de datos de test
x_test = pd.read_csv('practica_X_test.csv', sep=';', index_col=0)

# 3. Aplicar los tratamientos en el mismo orden que en entrenamiento
x_test_tratado = selector_varianza.transform(x_test)
x_test_discretizado = discretizador.transform(x_test_tratado)

# 4. Predecir las etiquetas
y_pred = modelo.predict(x_test_discretizado)

# 5. Guardar las etiquetas en un CSV con el mismo formato que practica_Y_train.csv
# Si tienes el diccionario de codificación, puedes invertirlo para obtener las etiquetas originales.
# Aquí se asume que las etiquetas son numéricas. Si necesitas mapear a texto, ajusta según tu codificación.
df_pred = pd.DataFrame(y_pred, index=x_test.index, columns=['Air_Quality'])
df_pred.to_csv('practica_Y_test.csv', sep=';')

print("Predicciones guardadas en practica_Y_test.csv")