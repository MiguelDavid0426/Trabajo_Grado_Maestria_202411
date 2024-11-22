# Codigo para consolidar la información y homologar las etiquetas
import os
import pandas as pd
import numpy as np
import json
import ast
import cv2
import re
from sklearn.model_selection import train_test_split
import mediapipe as mp

# Directorio donde se encuentran los videos
videos_dir = 'D:/OneDrive - Pontificia Universidad Javeriana/Codigo/Videos_tratados_V3'
videos = os.listdir(videos_dir)[35:36]

# Inicializar una lista para almacenar los DataFrames
dfs = []

# Mapeo de etiquetas a números
mapeo_etiquetas = {
    'Triste': 1,
    'Enojado': 2,
    'Neutral': 3,
    'Sorprendido': 4,
    'Feliz': 5,
    'No_Disponible': 0
}

# Función para homologar etiquetas solo en los elementos 0, 1 y 3
def homologar_etiquetas(x):
    x = ast.literal_eval(x)
    return [mapeo_etiquetas.get(etiqueta, etiqueta) if i in [0, 1, 2, 3] else etiqueta for i, etiqueta in enumerate(x)]

# Función para calcular la similitud del histograma entre dos matrices
def calcular_similitud_histograma(mat1, mat2, threshold=0.998):
    hist1 = cv2.calcHist([mat1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([mat2], [0], None, [256], [0, 256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return similarity > threshold

# Función para colorear las áreas de las etiquetas con negro
def colorear_areas_con_etiquetas(frame, etiquetas):   
    # Anchos basados en el tamaño de las etiquetas
    w = w1 = w2 = w4 = 205
    
    x1, y1 = 0, 0  # Esquina superior izquierda
    x2, y2 = frame.shape[1] - w, 0  # Esquina superior derecha
    x3, y3 = 0, frame.shape[0] - 25  # Esquina inferior izquierda
    x4, y4 = frame.shape[1] - w, frame.shape[0] - 70  # Esquina inferior derecha

    h1 = h2 = h4 = 50
    h3 = 25
    # Colorear las áreas con negro
    frame[y1:y1 + h1, x1:x1 + w1] = 0
    frame[y2:y2 + h2, x2:x2 + w2] = 0
    frame[y3:y3 + h3, x3:] = 0  # La tercera etiqueta se extiende hasta el borde derecho
    frame[y4:y4 + h4, x4:x4 + w4] = 0
    return frame

# Procesar cada archivo de video
for video in videos:
    if video.endswith('.h5'):
        archivo_h5 = os.path.join(videos_dir, video)
        # Leer el archivo HDF5
        df = pd.read_hdf(archivo_h5, key='datos')
        print("El tamaño del DF antes de eliminar duplicados es:", df.shape)
        
        df['Etiqueta_Homologada'] = df.iloc[:, 1].apply(homologar_etiquetas)
                
        df_homologado = df.copy()
        
        # Eliminar filas duplicadas dentro de cada archivo usando histograma de imagen
        unique_indices = []
        for i, row1 in df.iterrows():
            is_duplicate = False
            for j in unique_indices:
                if calcular_similitud_histograma(row1.iloc[0], df.iloc[j, 0]):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_indices.append(i)

        df_homologado = df_homologado.iloc[unique_indices]
        
        # Colorear las áreas con etiquetas con negro en cada frame
        for idx in range(df_homologado.shape[0]):
            frame = df_homologado.iloc[idx, 0]  # Suponiendo que el frame es la primera columna
            etiquetas = df_homologado.iloc[idx, 1]  # Suponiendo que las etiquetas están en la segunda columna
            frame = colorear_areas_con_etiquetas(frame, etiquetas)
            df_homologado.iloc[idx, 0] = frame
 
        # Filtrar los datos para eliminar las filas donde el tercer elemento es 'No_Disponible'
        #df_homologado = df_homologado[df_homologado['columna_homologada'].apply(lambda x: x[2] != 0)]
        #df_homologado = df_homologado[df_homologado['Var_Respuestas'].apply(lambda x: x[0] != 0)]
        
        # Agregar la nueva columna según las reglas especificadas
        # lista_b = []
        # for etiqueta in df_homologado['columna_homologada']:
        #     if etiqueta[0] > 0 and etiqueta[1] > 0 and etiqueta[3] == 0:
        #         lista_b.append([etiqueta[1], etiqueta[0], etiqueta[2]])
        #     elif etiqueta[1] > 0 and etiqueta[3] > 0 and etiqueta[0] == 0:
        #         lista_b.append([etiqueta[1], etiqueta[3], etiqueta[2]])
        #     elif etiqueta[0] > 0 and etiqueta[1] > 0 and etiqueta[3] > 0:
        #         lista_b.append([etiqueta[1], etiqueta[3], etiqueta[2]])
        #     elif etiqueta[0] > 0 and etiqueta[3] > 0 and etiqueta[1] == 0:
        #         lista_b.append([etiqueta[0], etiqueta[3], etiqueta[2]])
        
        # df_homologado['Var_Respuestas'] = lista_b
        # Añadir el DataFrame procesado a la lista
        print("El tamaño del DF después de eliminar duplicados es:", df_homologado.shape)

        dfs.append(df_homologado)
        print("Ya quedó el video:", video)
        print("---------------------------------")
        
# Concatenar todos los DataFrames en uno solo
df_consolidado = pd.concat(dfs, ignore_index=True)
# Conservar solo las dos primeras variables

df_consolidado = df_consolidado[['Frame', 'Etiqueta', 'Etiqueta_Homologada']]
#%%
import cv2
import numpy as np
from deepface import DeepFace
import h5py

# Analizar la imagen para clasificar la emoción
index = 110
imagen = df_consolidado.iloc[index, 0]  # Suponiendo que 'Frame' es la primera columna
print(df_consolidado.iloc[index, 1])
print(df_consolidado.iloc[index, 2])

# Verificar el tipo de la imagen y procesarla adecuadamente
if isinstance(imagen, np.ndarray):
    # Asegúrate de que la imagen esté en formato BGR (si está en escala de grises, conviértela a BGR)
    if imagen.ndim == 2:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
else:
    raise ValueError("La imagen no es un numpy array")

# Analizar la imagen para clasificar la emoción
analysis = DeepFace.analyze(imagen, actions=['emotion'])

# Revisar la estructura de 'analysis'
print(analysis)

# Acceder a la emoción dominante correctamente
if isinstance(analysis, list) and len(analysis) > 0:
    analysis = analysis[0]  # Acceder al primer elemento si es una lista

predicted_emotion = analysis["dominant_emotion"] ### Modelo VGG-Face
print(f'Predicted emotion: {predicted_emotion}')

# Añadir la etiqueta resultante en la esquina superior derecha de la imagen
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(imagen, 
            predicted_emotion, 
            (10, 30), 
            font, 
            1, 
            (0, 255, 0), 
            2, 
            cv2.LINE_AA)

# Mostrar la imagen con OpenCV
cv2.imshow('Imagen con Emoción', imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()
#%%

import cv2
import numpy as np
from deepface import DeepFace

# Función para preprocesar la imagen y analizar la emoción
def analyze_emotion(frame):
    # Convertir a BGR si es necesario
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    
    # Analizar la imagen para clasificar la emoción
    analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

    # Revisar la estructura de 'analysis' y acceder a la emoción dominante
    if isinstance(analysis, list) and len(analysis) > 0:
        analysis = analysis[0]  # Acceder al primer elemento si es una lista

    predicted_emotion = analysis.get("dominant_emotion", "No emotion detected")
    return predicted_emotion

# Inicializar la cámara
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se puede abrir la cámara.")
    exit()

while True:
    # Capturar fotograma por fotograma
    ret, frame = cap.read()
    if not ret:
        print("Error: No se puede capturar el fotograma.")
        break

    # Analizar la emoción en el fotograma
    predicted_emotion = analyze_emotion(frame)
    print(f'Predicted emotion: {predicted_emotion}')

    # Añadir la etiqueta resultante en la esquina superior derecha de la imagen
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, 
                predicted_emotion, 
                (10, 30), 
                font, 
                1, 
                (0, 255, 0), 
                2, 
                cv2.LINE_AA)

    # Mostrar la imagen con la etiqueta
    cv2.imshow('Video con Emoción', frame)

    # Detener el script con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
