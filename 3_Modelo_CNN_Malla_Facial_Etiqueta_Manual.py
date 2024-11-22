import os
import tensorflow as tf
from tensorflow.python.client import device_lib
import pandas as pd
import numpy as np
import json
import cv2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import ast
import matplotlib.pyplot as plt
import random

def check_gpu():
    print("TensorFlow version:", tf.__version__)

    # Verificar si TensorFlow está construido con soporte para CUDA
    is_built_with_cuda = tf.test.is_built_with_cuda()
    print("Built with CUDA:", is_built_with_cuda)
    
    if is_built_with_cuda:
        build_info = tf.sysconfig.get_build_info()
        cuda_version = build_info.get("cuda_version", "Not available")
        cudnn_version = build_info.get("cudnn_version", "Not available")
        print("CUDA version:", cuda_version)
        print("cuDNN version:", cudnn_version)
    else:
        print("TensorFlow no está construido con soporte para CUDA. Verifica la instalación de CUDA/cuDNN.")

    # Listar GPUs disponibles
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("GPUs available:")
        for gpu in gpus:
            print(gpu)
    else:
        print("No GPUs found.")
    
    # Listar todos los dispositivos locales
    print("Local devices:")
    devices = device_lib.list_local_devices()
    for device in devices:
        print(device)

def connect_to_gpu():
    try:
        # Intentar configurar la GPU para el uso de TensorFlow
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Usar solo la primera GPU
                tf.config.experimental.set_memory_growth(gpus[0], True)
                tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
                print("Successfully connected to GPU.")
                return True
            except RuntimeError as e:
                print("Error during GPU setup:", e)
                return False
        else:
            print("No GPUs found for connection.")
            return False
    except Exception as e:
        print("An unexpected error occurred:", e)
        return False

def main():
    # Habilitar el registro detallado
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    check_gpu()
    
    if not connect_to_gpu():
        print("Failed to connect to GPU. Please check the configuration and compatibility.")

if __name__ == "__main__":
    main()

# Directorio donde se encuentran los videos
videos_dir = 'D:/OneDrive - Pontificia Universidad Javeriana/Codigo/3_Datos_Manual_CNN'
videos = os.listdir(videos_dir)

# Lista para almacenar los DataFrames cargados
total_dfs = []

# Mapeo de etiquetas a números
mapeo_etiquetas = {
    'Triste': 1,
    'Enojado': 2,
    'Neutral': 3,
    'Sorprendido': 4,
    'Feliz': 5,
    'No_Disponible': 0,
    'fear': 'fear',
    'sad': 1,
    'happy': 5,
    'neutral':3,
    'surprise':4,
    'angry': 2,
    'disgust':'disgust',
    'Sad':1,
    'Surprise':4,
    'Happy':5,
    'Fear':'Fear',
    'Angry':2,
    'Disgust':'Disgust'    
}

# Función para homologar una lista de etiquetas
def homologar_etiquetas_lista(x):
    x = ast.literal_eval(x)
    return [mapeo_etiquetas.get(etiqueta, etiqueta) if i in [0, 1, 2, 3] else etiqueta for i, etiqueta in enumerate(x)] 
 
# Función para homologar una etiqueta individual
def homologar_etiquetas_individual(etiqueta):   
    # Homologamos directamente
    return mapeo_etiquetas.get(etiqueta, etiqueta)

# Función para deserializar JSON y manejar valores NaN
def safe_json_loads(x):
    if isinstance(x, str):
        return json.loads(x)
    return x

# Función para agregar la dimensión del canal a las imágenes y manejar valores NaN
def safe_expand_dims(x):
    if isinstance(x, str):
        return np.expand_dims(np.array(json.loads(x), dtype=np.uint8), axis=-1)
    return x

# Iterar sobre cada video en el directorio
for video in videos:
    video_path = os.path.join(videos_dir, video)
    segment_df = pd.read_csv(video_path)

    # Deserializar las columnas y agregar la dimensión del canal a las imágenes
    segment_df['Frame'] = segment_df['Frame'].apply(safe_expand_dims)
    segment_df['Etiqueta'] = segment_df['Etiqueta'].apply(safe_json_loads)
    segment_df['Etiqueta_Homologada'] = segment_df['Etiqueta_Homologada'].apply(safe_json_loads)  
    segment_df['Puntos_Malla'] = segment_df['Puntos_Malla'].apply(safe_json_loads)

    total_dfs.append(segment_df)
    print(f'Base' , str(video), 'cargado.')

# Concatenar los DataFrames de train en uno solo
total_df_consolidado = pd.concat(total_dfs, ignore_index=True)

total_df_consolidado['Etiqueta_Homologada'] = total_df_consolidado.iloc[:, 1].apply(homologar_etiquetas_lista)
total_df_consolidado['Etiqueta_DeepFace_Homo'] = total_df_consolidado.iloc[:, 8].apply(homologar_etiquetas_individual)
total_df_consolidado['Etiqueta_Fer2013_Homo'] = total_df_consolidado.iloc[:, 9].apply(homologar_etiquetas_individual)
total_df_consolidado['Etiqueta_Modelo_Homo'] = total_df_consolidado.iloc[:, 10].apply(homologar_etiquetas_individual)

#total_df_consolidado = total_df_consolidado[total_df_consolidado['Etiqueta_Homologada'].apply(lambda x: x[0] != 0)]
print("Tamaño del conjunto de entrenamiento consolidado:", total_df_consolidado.shape)

# Verificar la forma de la imagen
print(f"Forma de una imagen: {total_df_consolidado['Frame'].iloc[0].shape}")

# Función para convertir a escala de grises y asegurarse de que tenga un solo canal
def convert_to_grayscale(image):
    # Asegurarse de que la imagen sea un array de NumPy
    image = np.array(image)

    # Verificar el número de canales y convertir a escala de grises si es necesario
    if len(image.shape) == 3:
        if image.shape[2] == 3:  # Imagen en color (RGB/BGR)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.shape[2] == 1:  # Imagen ya en escala de grises (un solo canal)
            gray_image = image[:, :, 0]  # Extraer el canal único
        else:
            raise ValueError(f"Imagen con un número de canales inesperado: {image.shape[2]}")
    elif len(image.shape) == 2:
        gray_image = image
    else:
        raise ValueError("Dimensiones de imagen inesperadas")

    # Asegurar que la imagen tenga un solo canal
    gray_image = np.expand_dims(gray_image, axis=-1)
    return gray_image

# Aplicar la conversión a la columna 'Frame'
total_df_consolidado['Frame'] = total_df_consolidado['Frame'].apply(lambda x: convert_to_grayscale(x))

# Verificar la forma de una imagen después del procesamiento
print(f"Forma de una imagen procesada: {total_df_consolidado['Frame'].iloc[0].shape}")

#%%
# Visualizar la imagen en la fila 'numero' del DataFrame
numero = 1102

imagen = total_df_consolidado.iloc[numero, 0]  # Suponiendo que 'Frame' es la primera columna
print(total_df_consolidado.iloc[numero, 1])
print(total_df_consolidado.iloc[numero, 2])

# Verificar que la imagen es un numpy array antes de visualizarla
if isinstance(imagen, np.ndarray):
    # Si la imagen tiene una dimensión extra, eliminarla
    if imagen.ndim == 4 and imagen.shape[3] == 1:
        imagen = np.squeeze(imagen, axis=3)
    
    # Verificar que la imagen tiene dimensiones válidas
    if imagen.ndim == 2 or (imagen.ndim == 3 and imagen.shape[2] in [1, 3]):
        # Convertir a tipo uint8 si no lo es
        if imagen.dtype != np.uint8:
            imagen = imagen.astype(np.uint8)
        # Asegurarse de que la imagen esté en el rango correcto de valores
        imagen = np.clip(imagen, 0, 255)
        # Visualizar la imagen    
        cv2.imshow('Imagen', imagen)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("La imagen tiene dimensiones no válidas")
else:
    print("La imagen no es un numpy array")
#%%
# Resumen del número de datos por etiqueta
Etiqueta_Manual_summary = total_df_consolidado['Etiqueta_Manual'].value_counts()
print("Etiqueta_Manual: ",Etiqueta_Manual_summary)

Etiqueta_Homologada_summary = total_df_consolidado['Etiqueta_Homologada'].apply(lambda x: x[0]).value_counts()
print("Etiqueta_Homologada: ",Etiqueta_Homologada_summary)

# Resumen del número de datos por etiqueta
Etiqueta_DeepFace_summary = total_df_consolidado['Etiqueta_DeepFace_Homo'].value_counts()
print("Etiqueta_DeepFace: ",Etiqueta_DeepFace_summary)

# Resumen del número de datos por etiqueta
Etiqueta_Fer2013_summary = total_df_consolidado['Etiqueta_Fer2013_Homo'].value_counts()
print("Etiqueta_Fer2013: ",Etiqueta_Fer2013_summary)

# Resumen del número de datos por etiqueta
Etiqueta_Modelo_summary = total_df_consolidado['Etiqueta_Modelo_Homo'].value_counts()
print("Etiqueta_Modelo: ",Etiqueta_Modelo_summary)
#%%
# Crear una matriz de confusión usando crosstab
confusion_matrix = pd.crosstab(
    total_df_consolidado['Etiqueta_Manual'], 
    total_df_consolidado['Etiqueta_Modelo_Homo'], 
    values=total_df_consolidado['Etiqueta_Manual'],  # O cualquier otra columna si es necesario
    aggfunc='count',  # Contamos las ocurrencias
    rownames=['Etiqueta_Manual'], 
    colnames=['Etiqueta_Modelo_Homo'], 
    dropna=False
)
# Reemplazar los valores NaN por ceros
confusion_matrix = confusion_matrix.fillna(0)

# Mostrar la matriz de confusión
print("\nMatriz de Confusión:\n", confusion_matrix)

# Total de etiquetas: suma de todos los valores en la matriz de confusión
total_etiquetas = confusion_matrix.values.sum()
print(total_etiquetas)

#%%
# Obtener nombres únicos de videos
nombres_videos = total_df_consolidado['Nombre_Video'].unique()

# Dividir los nombres de videos en 70% para entrenamiento y 30% para prueba
train_videos = pd.Series(nombres_videos).sample(frac=0.6, random_state=42)
test_videos = pd.Series(nombres_videos).drop(train_videos.index)

# Filtrar los DataFrames usando los nombres de videos
train_df = total_df_consolidado[total_df_consolidado['Nombre_Video'].isin(train_videos)]
test_df = total_df_consolidado[total_df_consolidado['Nombre_Video'].isin(test_videos)]

# Mostrar el tamaño de los conjuntos
print("Tamaño del conjunto de entrenamiento:", train_df.shape)
print("Tamaño del conjunto de test:", test_df.shape)

# Resumen del número de datos por etiqueta   después del muestreo estratificado para train
train_label_summary = train_df['Etiqueta_Manual'].value_counts()
print("Resumen del número de datos por etiqueta en el conjunto de entrenamiento:")
print(train_label_summary)

# Resumen del número de datos por etiqueta después del muestreo estratificado para test
test_label_summary = test_df['Etiqueta_Manual'].value_counts()
print("Resumen del número de datos por etiqueta en el conjunto de test:")
print(test_label_summary)

# Determinar el número deseado de muestras por etiqueta (puedes ajustar este valor)
num_samples_train = 700
num_samples_train_cambios = 260
num_samples_test = 320

# Muestreo Estratificado para train
train_df_balanced = pd.DataFrame()
train_df_balanced_cambios = pd.DataFrame()

for etiqueta in train_df['Etiqueta_Manual'].unique():
    # Submuestreo (reducción de muestras de la clase mayoritaria)
    etiqueta_samples = train_df[train_df['Etiqueta_Manual'] == etiqueta]
    if len(etiqueta_samples) > num_samples_train:
        etiqueta_samples = etiqueta_samples.sample(num_samples_train, random_state=42)
    else:
        # Sobremuestreo (aumento de muestras de la clase minoritaria)
        etiqueta_samples = etiqueta_samples.sample(num_samples_train, replace=True, random_state=42)
    
    train_df_balanced = pd.concat([train_df_balanced, etiqueta_samples])

print("Tamaño del conjunto de entrenamiento después del balanceo:", train_df_balanced.shape)

for etiqueta in train_df['Etiqueta_Manual'].unique():
    # Submuestreo (reducción de muestras de la clase mayoritaria)
    etiqueta_samples = train_df[train_df['Etiqueta_Manual'] == etiqueta]
    if len(etiqueta_samples) > num_samples_train_cambios:
        etiqueta_samples = etiqueta_samples.sample(num_samples_train_cambios, random_state=42)
    else:
        # Sobremuestreo (aumento de muestras de la clase minoritaria)
        etiqueta_samples = etiqueta_samples.sample(num_samples_train_cambios, replace=True, random_state=42)
    
    train_df_balanced_cambios = pd.concat([train_df_balanced_cambios, etiqueta_samples])

print("Tamaño del conjunto de entrenamiento después del balanceo:", train_df_balanced_cambios.shape)

# Muestreo estratificado para prueba
test_df_balanced = pd.DataFrame()

for etiqueta in test_df['Etiqueta_Manual'].unique():
    # Submuestreo (reducción de muestras de la clase mayoritaria)
    etiqueta_samples = test_df[test_df['Etiqueta_Manual'] == etiqueta]
    if len(etiqueta_samples) > num_samples_test:
        etiqueta_samples = etiqueta_samples.sample(num_samples_test, random_state=42)
    else:
        # Sobremuestreo (aumento de muestras de la clase minoritaria)
        etiqueta_samples = etiqueta_samples.sample(num_samples_test, replace=True, random_state=42)
    
    test_df_balanced = pd.concat([test_df_balanced, etiqueta_samples])

print("\nTamaño del conjunto de prueba después del balanceo:", test_df_balanced.shape)

# Resumen del número de datos por etiqueta   después del muestreo estratificado para train
train_label_summary_balanced = train_df_balanced['Etiqueta_Manual'].value_counts()
print("Resumen del número de datos por etiqueta en el conjunto de entrenamiento:")
print(train_label_summary_balanced)

# Resumen del número de datos por etiqueta   después del muestreo estratificado para train
train_label_summary_balanced = train_df_balanced_cambios['Etiqueta_Manual'].value_counts()
print("Resumen del número de datos por etiqueta en el conjunto de entrenamiento cambio:")
print(train_label_summary_balanced)

# Resumen del número de datos por etiqueta en el conjunto de prueba después del balanceo
test_label_summary_balanced = test_df_balanced['Etiqueta_Manual'].value_counts()
print("Resumen del número de datos por etiqueta en el conjunto de prueba después del balanceo:")
print(test_label_summary_balanced)

# Función personalizada para aplicar rotación aleatoria entre -50 y 50 grados y volteo horizontal
def random_rotation_and_flip(img):
    # Rotación aleatoria entre 15 y 35 grados hacia derecha o izquierda
    angle = random.uniform(15, 35)
    angle = angle if random.choice([True, False]) else -angle
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Aplicar la rotación con bordes negros
    rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    # Volteo horizontal aleatorio
    if random.choice([True, False]):
        rotated = cv2.flip(rotated, 1)
    
    return rotated

# Para almacenar las nuevas imágenes y etiquetas
augmented_images = []
augmented_labels = []

# Iterar sobre el conjunto seleccionado y aplicar la rotación y volteo
for idx, row in train_df_balanced_cambios.iterrows():
    img = row['Frame']
    label = row['Etiqueta_Manual']

    # Convertir a array de NumPy si es necesario y asegurarse de que tiene tres dimensiones
    if isinstance(img, list):
        img = np.array(img, dtype=np.float32)
    if img.ndim == 2:  # Si la imagen es 2D (256, 256), expandir para tener (256, 256, 1)
        img = np.expand_dims(img, axis=-1)

    # Aplicar rotación y volteo personalizados
    augmented_img = random_rotation_and_flip(img)

    # Asegurar que la imagen aumentada tiene tres dimensiones (256, 256, 1)
    if augmented_img.ndim == 2:
        augmented_img = np.expand_dims(augmented_img, axis=-1)

    # Agregar la imagen aumentada y la etiqueta
    augmented_images.append(augmented_img)
    augmented_labels.append(label)

# Crear un nuevo DataFrame con las imágenes y etiquetas aumentadas
augmented_df = pd.DataFrame({
    'Frame': [img for img in augmented_images],
    'Etiqueta_Manual': augmented_labels
})

# Combinar el DataFrame original con el DataFrame de aumentación
train_df_augmented = pd.concat([train_df_balanced[['Frame', 'Etiqueta_Manual']], augmented_df], ignore_index=True)

# Resumen del número de datos por etiqueta en el conjunto después de la aumentación
train_label_summary_augmented = train_df_augmented['Etiqueta_Manual'].value_counts()
print("Resumen del número de datos por etiqueta en el conjunto después de la aumentación:")
print(train_label_summary_augmented)

#### Modelo para reconocer el estado emocional de la persona ####
# Configurar el crecimiento de la memoria de la GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print("Error al configurar la memoria de la GPU:", e)

#%%
#### MODELO
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch
import datetime

# Obtener la fecha actual en formato aaaammdd
fecha_actual = datetime.datetime.now().strftime('%Y%m%d')
# Definir el directorio con la fecha
directory = f'my_dir_{fecha_actual}'

# Define una clase de callback personalizada para imprimir métricas después de cada época
class PrintMetricsCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"\nEpoch {epoch+1}/{self.params['epochs']}")
        print(f" - loss: {logs['loss']:.4f} - accuracy: {logs['accuracy']:.4f}", end="")
        print(f" - val_loss: {logs['val_loss']:.4f} - val_accuracy: {logs['val_accuracy']:.4f}")

# Define la clase del modelo CNN HyperModel
class CNNHyperModel(HyperModel):
    def build(self, hp):
        model = models.Sequential()
        model.add(layers.Input(shape=(256, 256, 1)))
        
        max_pool_count = 0  # Contador de capas de MaxPooling
        
        for i in range(hp.Int('num_conv_layers', 6, 12)):
            
            model.add(layers.Conv2D(filters=hp.Int(f'filters_{i}', 8, 472, step=16),
                                    kernel_size=(3,3),
                                    activation='relu',
                                    padding='SAME'))
            model.add(layers.BatchNormalization())
            
            # Dropout adicional en las capas convolucionales
            model.add(layers.Dropout(rate=hp.Float(f'dropout_conv_{i}', 0.1, 0.5, step=0.1)))
            
            pool_size_choice = hp.Choice(f'pool_size_{i}', [2])
            pool_size = (pool_size_choice, pool_size_choice)
            
            # Selección entre MaxPooling y AveragePooling
            pool_type = hp.Choice(f'pool_type_{i}', ['MaxPooling', 'AveragePooling'])
            if pool_type == 'MaxPooling' and max_pool_count < 4:
                model.add(layers.MaxPooling2D(pool_size=pool_size))
                max_pool_count += 1
            elif pool_type == 'AveragePooling' and max_pool_count < 4:
                model.add(layers.AveragePooling2D(pool_size=pool_size))
                max_pool_count += 1
            
            # Verificar si detener el pooling
            if model.output_shape[1] <= pool_size_choice or model.output_shape[2] <= pool_size_choice:
                break
        
        # Opción para usar Global Average Pooling o Global Max Pooling en lugar de Flatten
        global_pooling_type = hp.Choice('global_pooling_type', ['GlobalAveragePooling', 'GlobalMaxPooling'])
        if global_pooling_type == 'GlobalAveragePooling':
            model.add(layers.GlobalAveragePooling2D())
        else:
            model.add(layers.GlobalMaxPooling2D())
             
        # Agregar capas densas con Dropout en la última capa
        num_dense_layers = hp.Int('num_dense_layers', 2, 4)
        for j in range(num_dense_layers):
            model.add(layers.Dense(units=hp.Int(f'dense_units_{j}', 8, 584, step=16), activation='relu'))
            if j < num_dense_layers - 1:
                model.add(layers.Dropout(rate=hp.Float(f'dropout_dense_{j}', 0.2, 0.4, step=0.1)))
            else:
                model.add(layers.Dropout(rate=0.3))  # Dropout en la última capa densa
        
        model.add(layers.Dense(5, activation='softmax'))
        
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

# Inicializar el tuner de Random Search
tuner = RandomSearch(
    CNNHyperModel(),
    objective='val_accuracy',
    max_trials=15,  # Aumentar el número de pruebas para explorar más combinaciones
    executions_per_trial=2,  # Ejecutar cada prueba dos veces para obtener resultados más estables
    directory= directory,
    project_name=f'emotion_recognition_{fecha_actual}', 
    overwrite=True,  # Sobrescribir resultados anteriores para evitar conflictos
)

# Generador personalizado para los datos de entrenamiento y validación
def custom_data_generator(df, batch_size):
    indices = df.index.tolist()
    while True:
        batch_indices = np.random.choice(indices, batch_size)
        batch_x = []
        batch_y = []
        for idx in batch_indices:
            img = df.loc[idx, 'Frame']
            label = df.loc[idx, 'Etiqueta_Manual']  # Ajustar las etiquetas para que empiecen en 0
            # Asegurarse de que 'label' es un valor único, no una Serie con índice
            if isinstance(label, pd.Series):
                label = label.values[0]  # Extraer el valor real
            
            label = int(label) - 1  # Ajustar para que las etiquetas empiecen en 0           
            # Asegurarse de que img sea un array de NumPy
            if isinstance(img, pd.Series):  # Si es una serie, convertirla
                img = img.iloc[0]
            # Si es una lista o estructura anidada, convertir a array
            if isinstance(img, list):
                img = np.array(img, dtype=np.float32)
            # Normalizar la imagen a valores entre 0 y 1
            img = img.astype(np.float32) / 255.0

            #img = np.expand_dims(img, axis=-1)
            # Convertir la imagen a escala de grises si es necesario
            if img.ndim == 3 and img.shape[2] == 3:
                img = np.mean(img, axis=-1, keepdims=True)  # Convertir a escala de grises
            batch_x.append(img)
            batch_y.append(label)
        
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        
        yield batch_x, batch_y

# Tamaño del lote
batch_size = 8

# Verificar el generador de datos
print("Verificando generador de datos...")
for batch_x, batch_y in custom_data_generator(train_df_augmented, batch_size):
    print(f"Tamaño del lote - X: {batch_x.shape}, Y: {batch_y.shape}")
    break
#%%
# Generadores de datos para entrenamiento y validación
train_generator = custom_data_generator(train_df_augmented, batch_size)
validation_generator = custom_data_generator(test_df_balanced, batch_size)

# Callbacks para Early Stopping, Model Checkpoint y el callback personalizado
callbacks = [
    EarlyStopping(monitor='val_loss', patience=18, restore_best_weights=True),
    ModelCheckpoint('best_emotion_model.h5', save_best_only=True),
    PrintMetricsCallback()
]

# Calcular el número de pasos por época y pasos de validación
steps_per_epoch = len(train_df_augmented) // batch_size
validation_steps = len(test_df_balanced) // batch_size
# Realizar la búsqueda de hiperparámetros
print("Iniciando la búsqueda de hiperparámetros...")
tuner.search(
    train_generator,
    validation_data=validation_generator,
    steps_per_epoch=steps_per_epoch,
    validation_steps= validation_steps,
    epochs=65,  # Aumentar el número de épocas para un mejor ajuste
    callbacks=callbacks,
    verbose=1
)

print("Mejores hiperparámetros encontrados:")
print(tuner.get_best_hyperparameters(num_trials=1)[0].values)

# Obtener los mejores hiperparámetros y el mejor modelo
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.hypermodel.build(best_hps)

print("Entrenando el mejor modelo...")

# Entrenar el mejor modelo
history = best_model.fit(
    train_generator,
    validation_data=validation_generator,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=65,
    callbacks=callbacks,
    verbose=1
)

print("Entrenamiento completado con el mejor modelo.")
# Mostrar la estructura del mejor modelo
best_model.summary()
#%%
import json
import matplotlib.pyplot as plt

# Reemplaza con la ruta correcta si el archivo no está en el mismo directorio
file_path = 'train_history_VF_2.json'

# Cargar el archivo JSON
with open(file_path, 'r') as file:
    history = json.load(file)

# Función para mostrar los resultados del entrenamiento
def plot_history(history):
    plt.figure(figsize=(12, 4))

    # Acceder a las métricas directamente desde el diccionario
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy over epochs')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over epochs')
    
    plt.show()

# Mostrar los resultados del entrenamiento
plot_history(history)
#%%
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

# Cargar el modelo sin compilar
best_model_VF_0 = load_model('Best_Emotion_Model_VF_0.h5', compile=False)
# Compilar el modelo manualmente
best_model_VF_0.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # Cargar el modelo sin compilar
# best_model_VF_1 = load_model('Best_Emotion_Model_VF_1.h5', compile=False)
# # Compilar el modelo manualmente
# best_model_VF_1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Cargar el modelo sin compilar
best_model_VF_2 = load_model('Best_Emotion_Model_VF_2.h5', compile=False)
# Compilar el modelo manualmente
best_model_VF_2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Mostrar la estructura del mejor modelo
#best_model.summary()
#%%
mapeo_etiquetas = {
    1: 'Triste',
    2: 'Enojado',
    3: 'Neutral',
    4: 'Sorprendido',
    5: 'Feliz'
}

# Función para obtener las predicciones y etiquetas verdaderas
def get_predictions_and_labels(generator, steps):
    predictions = []
    true_labels = []
    for _ in range(steps):
        batch_x, batch_y = next(generator)
        predictions.extend(np.argmax(best_model.predict(batch_x), axis=1) + 1)  # Ajustar las predicciones sumando 1
        true_labels.extend(batch_y + 1)  # Ajustar las etiquetas sumando 1
    return np.array(predictions), np.array(true_labels)

# Obtener generador de datos para los datos de prueba
batch_size_test = 32
test_generator = custom_data_generator(test_df_balanced, batch_size_test)
test_steps = len(test_df_balanced) // batch_size_test

# Obtener predicciones y etiquetas verdaderas para datos de prueba
test_predictions, test_labels = get_predictions_and_labels(test_generator, steps=test_steps)

# Función para mostrar la matriz de confusión con etiquetas mapeadas
def plot_confusion_matrix(labels, predictions, title, label_mapping):
    # Crear la matriz de confusión
    cm = confusion_matrix(labels, predictions)
    
    # Configurar la figura
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    
    # Obtener las etiquetas mapeadas para los ejes x e y
    tick_labels = [label_mapping[label] for label in sorted(label_mapping.keys())]
    
    # Configurar etiquetas de ejes y título
    plt.xticks(np.arange(len(tick_labels)) + 0.5, tick_labels, rotation=45)
    plt.yticks(np.arange(len(tick_labels)) + 0.5, tick_labels, rotation=0)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(title)
    
    # Mostrar la matriz de confusión
    plt.show()

# Matriz de confusión para datos de prueba con etiquetas mapeadas
plot_confusion_matrix(test_labels, test_predictions, title='Matriz de Confusión - Datos de Test', label_mapping=mapeo_etiquetas)
#%%
mapeo_etiquetas = {
    1: 'Triste',
    2: 'Enojado',
    3: 'Neutral',
    4: 'Sorprendido',
    5: 'Feliz'
}

# Función para obtener las predicciones y etiquetas verdaderas
def get_predictions_and_labels(generator, steps):
    predictions = []
    true_labels = []
    for _ in range(steps):
        batch_x, batch_y = next(generator)
        predictions.extend(np.argmax(best_model.predict(batch_x), axis=1) + 1)  # Ajustar las predicciones sumando 1
        true_labels.extend(batch_y + 1)  # Ajustar las etiquetas sumando 1
    return np.array(predictions), np.array(true_labels)

# Obtener generador de datos para los datos de prueba
batch_size_test = 32
test_generator = custom_data_generator(train_df_augmented, batch_size_test)
test_steps = len(train_df_augmented) // batch_size_test

# Obtener predicciones y etiquetas verdaderas para datos de prueba
test_predictions, test_labels = get_predictions_and_labels(test_generator, steps=test_steps)

# Función para mostrar la matriz de confusión con etiquetas mapeadas
def plot_confusion_matrix(labels, predictions, title, label_mapping):
    # Crear la matriz de confusión
    cm = confusion_matrix(labels, predictions)
    
    # Configurar la figura
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    
    # Obtener las etiquetas mapeadas para los ejes x e y
    tick_labels = [label_mapping[label] for label in sorted(label_mapping.keys())]
    
    # Configurar etiquetas de ejes y título
    plt.xticks(np.arange(len(tick_labels)) + 0.5, tick_labels, rotation=45)
    plt.yticks(np.arange(len(tick_labels)) + 0.5, tick_labels, rotation=0)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(title)
    
    # Mostrar la matriz de confusión
    plt.show()

# Matriz de confusión para datos de prueba con etiquetas mapeadas
plot_confusion_matrix(test_labels, test_predictions, title='Matriz de Confusión - Datos de Train', label_mapping=mapeo_etiquetas)

train_df_augmented, test_df_balanced
#%%
#Matriz confusión combinando modelos
# Mapeo de etiquetas
mapeo_etiquetas = {
    1: 'Triste',
    2: 'Enojado',
    3: 'Neutral',
    4: 'Sorprendido',
    5: 'Feliz'
}

# Función para integrar las predicciones de los dos modelos
def integrar_predicciones(batch_x):
    predicciones_combinadas = []
    for img in batch_x:
        # Realizar predicciones con ambos modelos
        pred_VF_0 = best_model_VF_0.predict(np.expand_dims(img, axis=0))
        pred_VF_2 = best_model_VF_2.predict(np.expand_dims(img, axis=0))

        # Obtener las clases predichas
        clase_VF_0 = np.argmax(pred_VF_0) + 1
        clase_VF_2 = np.argmax(pred_VF_2) + 1

        # Aplicar las reglas de decisión
        if clase_VF_0 == 2 or clase_VF_2 == 2:  # Si cualquiera predice Enojado
            predicciones_combinadas.append(2)
        elif clase_VF_0 in [1, 3]:  # Si VF_0 predice Triste o Neutral
            predicciones_combinadas.append(clase_VF_0)
        elif clase_VF_0 == 5 or clase_VF_2 == 5:  # Si cualquiera predice Feliz
            predicciones_combinadas.append(5)
        elif clase_VF_2 == 4:  # Si VF_2 predice Sorprendido
            predicciones_combinadas.append(4)
        
        else:
            # Si no se cumple ninguna regla, usar predicción de VF_2 por defecto
            predicciones_combinadas.append(clase_VF_2)
    return predicciones_combinadas

# Función para obtener las predicciones combinadas y etiquetas verdaderas
def get_predictions_and_labels(generator, steps):
    predictions = []
    true_labels = []
    for _ in range(steps):
        batch_x, batch_y = next(generator)
        predictions.extend(integrar_predicciones(batch_x))  # Usar las reglas de decisión
        true_labels.extend(batch_y + 1)  # Ajustar etiquetas para que empiecen en 1
    return np.array(predictions), np.array(true_labels)

# Obtener generador de datos para los datos de prueba
batch_size_test = 32
test_generator = custom_data_generator(train_df_augmented, batch_size_test)
test_steps = len(train_df_augmented) // batch_size_test

# Obtener predicciones y etiquetas verdaderas para datos de prueba
test_predictions, test_labels = get_predictions_and_labels(test_generator, steps=test_steps)

# Función para mostrar la matriz de confusión con etiquetas mapeadas
def plot_confusion_matrix(labels, predictions, title, label_mapping):
    # Crear la matriz de confusión
    cm = confusion_matrix(labels, predictions)
    
    # Configurar la figura
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    
    # Obtener las etiquetas mapeadas para los ejes x e y
    tick_labels = [label_mapping[label] for label in sorted(label_mapping.keys())]
    
    # Configurar etiquetas de ejes y título
    plt.xticks(np.arange(len(tick_labels)) + 0.5, tick_labels, rotation=45)
    plt.yticks(np.arange(len(tick_labels)) + 0.5, tick_labels, rotation=0)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(title)
    
    # Mostrar la matriz de confusión
    plt.show()

# Matriz de confusión para datos de prueba con etiquetas mapeadas
plot_confusion_matrix(test_labels, test_predictions, title='Matriz de Confusión - Datos de Train (Combinados)', label_mapping=mapeo_etiquetas)
#%%
# Video con camara utilizando el ensamble 
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

# Cargar los modelos sin compilar
best_model_VF_0 = load_model('Best_Emotion_Model_VF_0.h5', compile=False)
best_model_VF_2 = load_model('Best_Emotion_Model_VF_2.h5', compile=False)

# Compilar los modelos manualmente
best_model_VF_0.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
best_model_VF_2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Mapeo de etiquetas
mapeo_etiquetas = {
    1: 'Triste',
    2: 'Enojado',
    3: 'Neutral',
    4: 'Sorprendido',
    5: 'Feliz'
}

# Inicializar Mediapipe para la malla facial
mpMallaFacial = mp.solutions.face_mesh
MallaFacial = mpMallaFacial.FaceMesh(max_num_faces=1)
mpDibujo = mp.solutions.drawing_utils
ConfDibu = mpDibujo.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=0)

# Función para recortar y redimensionar imágenes basado en los puntos de la malla facial
def recortar_y_redimensionar(image, landmarks, output_size=(256, 256), margin=0.2):
    landmarks = np.array(landmarks)
    x_min = np.min(landmarks[:, 0]) * image.shape[1]
    y_min = np.min(landmarks[:, 1]) * image.shape[0]
    x_max = np.max(landmarks[:, 0]) * image.shape[1]
    y_max = np.max(landmarks[:, 1]) * image.shape[0]
    width = x_max - x_min
    height = y_max - y_min
    x_min = max(int(x_min - margin * width), 0)
    y_min = max(int(y_min - margin * height), 0)
    x_max = min(int(x_max + margin * width), image.shape[1])
    y_max = min(int(y_max + margin * height), image.shape[0])
    cropped_image = image[y_min:y_max, x_min:x_max]
    resized_image = cv2.resize(cropped_image, output_size)
    return resized_image

# Función para extraer puntos de la malla facial
def extraer_puntos_malla(resultados):
    puntos = []
    if resultados.multi_face_landmarks:
        for rostro in resultados.multi_face_landmarks:
            for punto in rostro.landmark:
                puntos.append((punto.x, punto.y, punto.z))
    return puntos

# Función para integrar las predicciones de los dos modelos
def integrar_predicciones(frame_expanded):
    pred_VF_0 = best_model_VF_0.predict(frame_expanded)
    pred_VF_2 = best_model_VF_2.predict(frame_expanded)

    # Obtener las clases predichas
    clase_VF_0 = np.argmax(pred_VF_0) + 1
    clase_VF_2 = np.argmax(pred_VF_2) + 1

    # Aplicar reglas de decisión
    if clase_VF_0 == 2 or clase_VF_2 == 2:
        return 2, mapeo_etiquetas[2]
    elif clase_VF_0 in [1, 3]:
        return clase_VF_0, mapeo_etiquetas[clase_VF_0]
    elif clase_VF_2 == 5:
        return 5, mapeo_etiquetas[5]
    elif clase_VF_2 == 4:
        return 4, mapeo_etiquetas[4]
    else:
        return clase_VF_2, mapeo_etiquetas[clase_VF_2]

# Inicializar la captura de video desde la cámara (0 es la cámara predeterminada)
cap = cv2.VideoCapture(0)

# Verificar si la cámara se abrió correctamente
if not cap.isOpened():
    print("Error al abrir la cámara")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Procesar la malla facial con mediapipe
    resultados = MallaFacial.process(frame)

    # Extraer puntos de la malla facial
    puntos_malla = extraer_puntos_malla(resultados)

    # Inicializar variables para las imágenes a mostrar
    imagen_recortada_redimensionada = None
    predicted_emotion_label = "No se detectaron puntos de malla facial"

    if puntos_malla:
        # Dibujar la malla facial sobre la imagen original
        imagen_malla_sobre_negro = np.zeros_like(frame, dtype=np.uint8)
        for rostro in resultados.multi_face_landmarks:
            mpDibujo.draw_landmarks(imagen_malla_sobre_negro, rostro, mpMallaFacial.FACEMESH_CONTOURS, ConfDibu, ConfDibu)

        # Recortar y redimensionar la imagen basada en los puntos de la malla facial
        imagen_recortada_redimensionada = recortar_y_redimensionar(imagen_malla_sobre_negro, puntos_malla)

        # Convertir la imagen a escala de grises si es necesario
        if imagen_recortada_redimensionada.shape[-1] == 3:
            imagen_recortada_redimensionada = cv2.cvtColor(imagen_recortada_redimensionada, cv2.COLOR_BGR2GRAY)

        # Preparar la imagen para la predicción del modelo
        frame_expanded = np.expand_dims(imagen_recortada_redimensionada, axis=-1)
        frame_expanded = np.expand_dims(frame_expanded, axis=0)
        frame_expanded = frame_expanded.astype(np.float32) / 255.0  # Normalización

        # Integrar las predicciones de los modelos
        predicted_emotion_index, predicted_emotion_label = integrar_predicciones(frame_expanded)

    # Mostrar la emoción predicha en el fotograma
    cv2.putText(frame, predicted_emotion_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Mostrar la imagen original con la etiqueta
    cv2.imshow("Imagen Original", frame)

    # Mostrar la imagen recortada y redimensionada
    if imagen_recortada_redimensionada is not None:
        cv2.imshow("Imagen Recortada y Redimensionada", imagen_recortada_redimensionada)

    # Presionar 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#%%
# Mapeo de etiquetas
mapeo_etiquetas = {
    1: 'Triste',
    2: 'Enojado',
    3: 'Neutral',
    4: 'Sorprendido',
    5: 'Feliz'
}

# Inicializar Mediapipe para la malla facial
mpMallaFacial = mp.solutions.face_mesh
MallaFacial = mpMallaFacial.FaceMesh(max_num_faces=1)
mpDibujo = mp.solutions.drawing_utils
ConfDibu = mpDibujo.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=0)

# Función para recortar y redimensionar imágenes basado en los puntos de la malla facial
def recortar_y_redimensionar(image, landmarks, output_size=(256, 256), margin=0.2):
    landmarks = np.array(landmarks)
    x_min = np.min(landmarks[:, 0]) * image.shape[1]
    y_min = np.min(landmarks[:, 1]) * image.shape[0]
    x_max = np.max(landmarks[:, 0]) * image.shape[1]
    y_max = np.max(landmarks[:, 1]) * image.shape[0]
    width = x_max - x_min
    height = y_max - y_min
    x_min = max(int(x_min - margin * width), 0)
    y_min = max(int(y_min - margin * height), 0)
    x_max = min(int(x_max + margin * width), image.shape[1])
    y_max = min(int(y_max + margin * height), image.shape[0])
    cropped_image = image[y_min:y_max, x_min:x_max]
    resized_image = cv2.resize(cropped_image, output_size)
    return resized_image

# Función para extraer puntos de la malla facial
def extraer_puntos_malla(resultados):
    puntos = []
    if resultados.multi_face_landmarks:
        for rostro in resultados.multi_face_landmarks:
            for punto in rostro.landmark:
                puntos.append((punto.x, punto.y, punto.z))
    return puntos

# Inicializar la captura de video desde la cámara (0 es la cámara predeterminada)
cap = cv2.VideoCapture(0)

# Verificar si la cámara se abrió correctamente
if not cap.isOpened():
    print("Error al abrir la cámara")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Procesar la malla facial con mediapipe
    resultados = MallaFacial.process(frame)

    # Extraer puntos de la malla facial
    puntos_malla = extraer_puntos_malla(resultados)

    # Inicializar variables para las imágenes a mostrar
    imagen_recortada_redimensionada = None
    predicted_emotion_label = "No se detectaron puntos de malla facial"

    if puntos_malla:
        # Dibujar la malla facial sobre la imagen original
        imagen_malla_sobre_negro = np.zeros_like(frame, dtype=np.uint8)
        for rostro in resultados.multi_face_landmarks:
            mpDibujo.draw_landmarks(imagen_malla_sobre_negro, rostro, mpMallaFacial.FACEMESH_CONTOURS, ConfDibu, ConfDibu)

        # Recortar y redimensionar la imagen basada en los puntos de la malla facial
        imagen_recortada_redimensionada = recortar_y_redimensionar(imagen_malla_sobre_negro, puntos_malla)

        # Convertir la imagen a escala de grises si es necesario
        if imagen_recortada_redimensionada.shape[-1] == 3:
            imagen_recortada_redimensionada = cv2.cvtColor(imagen_recortada_redimensionada, cv2.COLOR_BGR2GRAY)

        # Preparar la imagen para la predicción del modelo
        frame_expanded = np.expand_dims(imagen_recortada_redimensionada, axis=-1)
        frame_expanded = np.expand_dims(frame_expanded, axis=0)
        frame_expanded = frame_expanded.astype(np.float32) / 255.0  # Normalización

        # Predecir la emoción
        emotion_prediction_VF_0 = best_model_VF_0.predict(frame_expanded)
        # emotion_prediction_VF_1 = best_model_VF_1.predict(frame_expanded)
        emotion_prediction_VF_2 = best_model_VF_2.predict(frame_expanded)
        # Obtener la emoción predicha
        predicted_emotion_index_VF_0 = np.argmax(emotion_prediction_VF_0) + 1
        # predicted_emotion_index_VF_1 = np.argmax(emotion_prediction_VF_1) + 1
        predicted_emotion_index_VF_2 = np.argmax(emotion_prediction_VF_2) + 1
        # Verificar si el índice predicho está en el mapeo de etiquetas
        if predicted_emotion_index_VF_0 in mapeo_etiquetas:
            predicted_emotion_label_VF_0 = mapeo_etiquetas[predicted_emotion_index_VF_0]
        else:
            predicted_emotion_label_VF_0 = 'N/A'
            print(f"Índice de emoción predicha fuera del rango: {predicted_emotion_index_VF_0}")
            
        # # Verificar si el índice predicho está en el mapeo de etiquetas
        # if predicted_emotion_index_VF_1 in mapeo_etiquetas:
        #     predicted_emotion_label_VF_1 = mapeo_etiquetas[predicted_emotion_index_VF_1]
        # else:
        #     predicted_emotion_label_VF_1 = 'N/A'
        #     print(f"Índice de emoción predicha fuera del rango: {predicted_emotion_index_VF_1}")
            
        # Verificar si el índice predicho está en el mapeo de etiquetas
        if predicted_emotion_index_VF_2 in mapeo_etiquetas:
            predicted_emotion_label_VF_2 = mapeo_etiquetas[predicted_emotion_index_VF_2]
        else:
            predicted_emotion_label_VF_2 = 'N/A'
            print(f"Índice de emoción predicha fuera del rango: {predicted_emotion_index_VF_2}")

        # Mostrar la emoción predicha en el fotograma
        cv2.putText(frame, predicted_emotion_label_VF_0, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        # Mostrar la emoción predicha en el fotograma
        # cv2.putText(frame, predicted_emotion_label_VF_1, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        # Mostrar la emoción predicha en el fotograma
        cv2.putText(frame, predicted_emotion_label_VF_2, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    else:
        predicted_emotion_label = "No se detectaron puntos de malla facial"

    # Mostrar la imagen original con la etiqueta
    cv2.imshow("Imagen Original", frame)

    # Mostrar la imagen recortada y redimensionada
    if imagen_recortada_redimensionada is not None:
        cv2.imshow("Imagen Recortada y Redimensionada", imagen_recortada_redimensionada)

    # Imprimir la etiqueta predicha
    print(predicted_emotion_label)

    # Presionar 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
#%%
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Cargar el modelo
#ruta_guardado_modelo = "D:/OneDrive - Pontificia Universidad Javeriana/Codigo/Datos_Modelo_RL/emotion_model"
#modelo_cargado = load_model(ruta_guardado_modelo) 

# Cargar el video
cap = cv2.VideoCapture('2024_04_25__16_45_43_Estudiante_Miguel_Benavides.avi')

mapeo_etiquetas = {
    1: 'Triste',
    2: 'Enojado',
    3: 'Neutral',
    4: 'Sorprendido',
    5: 'Feliz'
}
# Definir los recortes
top_crop = 25  # Recorte en la parte superior
bottom_crop = 24  # Recorte en la parte inferior
left_crop = 30  # Recorte en la parte izquierda
right_crop = 30  # Recorte en la parte derecha

# Función para recortar y redimensionar imágenes
def recortar_y_redimensionar(image, top_crop, bottom_crop, left_crop, right_crop, new_size=(180, 240)):
    # Recortar las áreas negras
    cropped_image = image[top_crop:-bottom_crop, left_crop:-right_crop]
    # Redimensionar la imagen
    resized_image = cv2.resize(cropped_image, new_size)
    return resized_image

# Verificar si el video se abrió correctamente
if not cap.isOpened():
    print("Error al abrir el archivo de video")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break    
        
    # Convertir a escala de grises y aplicar recorte y redimensionamiento
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_cropped_resized = recortar_y_redimensionar(frame_gray, top_crop, bottom_crop, left_crop, right_crop, new_size=(240, 180))
    
    # Preparar la imagen para la predicción del modelo
    frame_expanded = np.expand_dims(frame_cropped_resized, axis=-1)
    frame_expanded = np.expand_dims(frame_expanded, axis=0)
    frame_expanded = frame_expanded.astype(np.float32) / 255.0  # Normalización
    
    # Predecir la emoción
    emotion_prediction = best_model.predict(frame_expanded)
    
    # Obtener la emoción predicha
    predicted_emotion_index = np.argmax(emotion_prediction)
    
    # Verificar si el índice predicho está en el mapeo de etiquetas
    if predicted_emotion_index in mapeo_etiquetas:
        predicted_emotion_label = mapeo_etiquetas[predicted_emotion_index]
    else:
        predicted_emotion_label = 'N/A'
        print(f"Índice de emoción predicha fuera del rango: {predicted_emotion_index}")
    
    # Mostrar la emoción predicha en el fotograma
    cv2.putText(frame, predicted_emotion_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Predicción: {emotion_prediction}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)  # Muestra el valor de la predicción
    
    cv2.imshow("Frame", frame)
    print(predicted_emotion_label)
    
    # Presionar 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()