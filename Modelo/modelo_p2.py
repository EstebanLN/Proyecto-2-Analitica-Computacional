#Importo las librerías
import numpy as np
import pandas as pd
import tensorflow as tf
import keras

#Importo los datos de la limpieza de datos 
df =pd.read_csv("Modelo\Clean_data.csv")

#print(df.head)
#print(df.shape)

df_dummies = pd.get_dummies(df)

#Separo los datos en entrenamiento, validación y prueba 

#Entrenamiento 
train = df.sample(frac=0.8, random_state=100)

#Validación 
test = df.drop(train.index)

#Prueba
val = train.sample(frac=0.2, random_state=100)

#Actualizo train quitando las filas que se fueron para validación 
train = train.drop(val.index)

#Verificar los tamaños 
print(train.shape) 
print(val.shape)    
print(test.shape)

#Calulo estadísticas para cada variable 

train.describe()

#Convierto DataFrame(Pandas) a dataset(Tensorflow)
def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("punt_global")  #me interesa esta 
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds

