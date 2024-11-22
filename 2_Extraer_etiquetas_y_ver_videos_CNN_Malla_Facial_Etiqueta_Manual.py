import os
import cv2
import time
import numpy as np
import pandas as pd
import pytesseract
from fuzzywuzzy import fuzz
import json
## Codigo para tomar cada video y extraer cada etiqueta, reconocerla y guardar cada video en un dataframe con formato H5

# Directorio donde se encuentran los videos
videos_dir = 'D:/OneDrive - Pontificia Universidad Javeriana/Codigo/Videos_Javeriana'
videos = os.listdir(videos_dir)[43:44]

# Configurar el idioma para el reconocimiento óptico de caracteres (OCR)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Ruta a tu instalación de Tesseract

# Leer datos de música
music_df = pd.read_csv("data_moods.csv")
music_df = music_df[['name', 'artist']]
music_df["cancion"] = music_df["artist"] + " " + music_df["name"]

# Función para cargar los fotogramas de un video
def load_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)
    cap.release()
    return frames

def obtener_etiquetas(frames, music_df):
    emociones = ['Enojado', 'Feliz', 'Neutral', 'Triste', 'Sorprendido']
    h = 0
    umbral_similitud = 55  # Porcentaje de similitud mínimo aceptable
    umbral_similitud_ = 85
    
    etiquetas_totales = []
    x1, y1, w1, h1 = 0, 0, 110, 50  # coordenadas etiquetas
    x2, y2, w2, h2 = 515, 0, 110, 50
    x3, y3, w3, h3 = 0, 455, 160, 25
    x4, y4, w4, h4 = 515, 410, 110, 50
    
    for frame in frames:
        roi_1 = frame[y1:y1+h1, x1:x1+w1]
        roi_2 = frame[y2:y2+h2, x2:x2+w2]
        roi_3 = frame[y3:y3+h3, x3:x3+w3]
        roi_4 = frame[y4:y4+h4, x4:x4+w4]
        
        text_1 = pytesseract.image_to_string(roi_1, lang='spa')
        text_2 = pytesseract.image_to_string(roi_2, lang='spa')
        text_3 = pytesseract.image_to_string(roi_3, lang='eng')
        text_4 = pytesseract.image_to_string(roi_4, lang='spa')
        
        similitudes_1 = [fuzz.partial_ratio(emocion, text_1) for emocion in emociones]
        if max(similitudes_1) >= umbral_similitud:
            text_1 = emociones[similitudes_1.index(max(similitudes_1))]
        else:
            text_1 = 'No_Disponible'
        
        similitudes_2 = [fuzz.partial_ratio(emocion, text_2) for emocion in emociones]
        if max(similitudes_2) >= umbral_similitud:
            text_2 = emociones[similitudes_2.index(max(similitudes_2))]
        else:
            text_2 = 'No_Disponible'
            
        for cancion in music_df["cancion"]:
            similitud = fuzz.partial_ratio(cancion.lower(), text_3.lower())
            if similitud >= umbral_similitud_:
                text_3 = cancion
                break
        else:
            text_3 = 'No_Disponible'

        similitudes_4 = [fuzz.partial_ratio(emocion, text_4) for emocion in emociones]
        if max(similitudes_4) >= umbral_similitud:
            text_4 = emociones[similitudes_4.index(max(similitudes_4))]
        else:
            text_4 = 'No_Disponible'
        
        etiquetas = [text_1, text_2, text_3, text_4]
        
        print(etiquetas)
        
        etiquetas_totales.append(etiquetas)
        
        # Mostrar las ROI y el video frame
        cv2.imshow("ROI 1", roi_1)
        cv2.imshow("ROI 2", roi_2)
        cv2.imshow("ROI 3", roi_3)
        cv2.imshow("ROI 4", roi_4)
        cv2.imshow("Frame", frame)
        
        # Esperar para permitir visualización, ajustar el tiempo según necesites
        cv2.waitKey(1)
        
        print(h)
        print("------------------------")
        h += 1
    
    cv2.destroyAllWindows()
    return etiquetas_totales

# Procesar cada video
for video_file in videos:
    frames_data = []
    labels_data = []
    
    start_time = time.time()
    video_path = os.path.join(videos_dir, video_file)
    frames = load_frames(video_path)
    labels = obtener_etiquetas(frames, music_df)
    
    end_time = time.time()
    execution_time = float(end_time - start_time) / 60
    print(video_file)
    print("El proceso tardó:", execution_time, "minutos")
            
    for frame, label in zip(frames, labels):
        if sum(tag != 'No_Disponible' for i, tag in enumerate(label) if i in (0, 1, 3)) >= 2:
            frames_data.append(frame)
            labels_data.append(label)

    df = pd.DataFrame({'Frame': frames_data, 'Etiqueta': labels_data})
    df['Frame'] = df['Frame'].apply(lambda x: np.array(x))
    df['Etiqueta'] = df['Etiqueta'].apply(lambda x: json.dumps(x))
 
    if not df.empty:
        nombre_guardar = video_file.replace('.avi', '')
        df.to_hdf(f'D:/OneDrive - Pontificia Universidad Javeriana/Codigo/2_Videos_tratados/{nombre_guardar}.h5', key="datos", mode="w")
#%%
# Codigo para consolidar la información, homologar las etiquetas, crear variables relevantes para identificar a la persona y el video, se extrae la malla 
# fecial, se califica la emoción manualmente y se guarda cada respectivo video
import os
import pandas as pd
import numpy as np
import json
import ast
import cv2
import re
from sklearn.model_selection import train_test_split
from IPython.display import clear_output
import mediapipe as mp
from deepface import DeepFace
from keras.models import load_model
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array

# Directorio donde se encuentran los videos
videos_dir = 'D:/OneDrive - Pontificia Universidad Javeriana/Codigo/2_Videos_tratados'
videos = os.listdir(videos_dir)[38:]
#%%
### Correr este si eres Christian 
#videos.reverse()
#videos = videos[11:]

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

# Función para extraer el nombre del video y la fecha y hora
def extraer_info_video(nombre_archivo):
    # Usar expresión regular para extraer el nombre entre 'Estudiante_' y '.h5'
    nombre_match = re.search(r'Estudiante_(.*?)\.h5$', nombre_archivo)
    nombre = nombre_match.group(1) if nombre_match else None
    
    # Usar expresión regular para extraer la fecha y hora entre '__' y '_estudiante'
    fecha_hora_match = re.search(r'__(.*?)_Estudiante_', nombre_archivo)
    fecha_hora = fecha_hora_match.group(1) if fecha_hora_match else None
    
    return nombre, fecha_hora

# Procesar cada archivo de video
for video in videos:
    if video.endswith('.h5'):
        archivo_h5 = os.path.join(videos_dir, video)
        # Leer el archivo HDF5
        df = pd.read_hdf(archivo_h5, key='datos')
        print("El tamaño del DF antes de eliminar duplicados es:", df.shape)
        
        df['Etiqueta_Homologada'] = df.iloc[:, 1].apply(homologar_etiquetas)
        
        # Crear una columna enumerando las filas desde el momento de cargar el video
        df['Indice_Video'] = np.arange(1, df.shape[0] + 1)

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

        # Resetear el índice del DataFrame
        df_homologado.reset_index(drop=True, inplace=True)

        # Colorear las áreas con etiquetas con negro en cada frame
        for idx in range(df_homologado.shape[0]):
            frame = df_homologado.iloc[idx, 0]  # Suponiendo que el frame es la primera columna
            etiquetas = df_homologado.iloc[idx, 1]  # Suponiendo que las etiquetas están en la segunda columna 
            frame_ = colorear_areas_con_etiquetas(frame, etiquetas)
            df_homologado.at[idx, 'Frame'] = frame_
        # Extraer el nombre del video y la fecha y hora
        nombre_video, fecha_hora = extraer_info_video(video)
        df_homologado['Nombre_Video'] = nombre_video
        df_homologado['Fecha_hora'] = fecha_hora
    
        # Filtrar los datos para eliminar las filas donde el tercer elemento es 'No_Disponible'
        #df_homologado = df_homologado[df_homologado['Var_Respuestas'].apply(lambda x: x[0] != 0)]
        
        # Añadir el DataFrame procesado a la lista
        print("El tamaño del DF después de eliminar duplicados es:", df_homologado.shape)
        print("Ya quedó el video:", video)
        df_consolidado = df_homologado[['Frame', 'Etiqueta', 'Etiqueta_Homologada', 'Nombre_Video', 'Fecha_hora', 'Indice_Video']]
        print("--------------------------------------------------------------------------")

        # Inicializar mediapipe para la malla facial
        mpMallaFacial = mp.solutions.face_mesh
        MallaFacial = mpMallaFacial.FaceMesh(max_num_faces=1)
        mpDibujo = mp.solutions.drawing_utils
        ConfDibu = mpDibujo.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=0)
        
        # Cargar el modelo fer2013_mini_XCEPTION
        model_path = 'Modelos_Pre_entrenados/fer2013_mini_XCEPTION.119-0.65.hdf5'
        model = load_model(model_path, compile=False)
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Cargar el modelo sin compilar
        best_model = load_model('Best_Emotion_Model_20240721.h5', compile=False)
        # Compilar el modelo manualmente
        best_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        # Mapeo de etiquetas
        mapeo_etiquetas = {
            1: 'Triste',
            2: 'Enojado',
            3: 'Neutral',
            4: 'Sorprendido',
            5: 'Feliz'
        }
        
        # Función para preprocesar la imagen según el modelo
        def preprocess_image(image, target_size=(48, 48)):
            image = cv2.resize(image, target_size)
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises
            image = image.astype('float32') / 255.0
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            return image
        
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
        
        # Crear nuevas columnas en el dataframe para almacenar las listas modificadas
        df_consolidado['Etiqueta_Manual'] = None
        df_consolidado['Puntos_Malla'] = None
        
        # Función para validar la entrada del usuario
        def validar_entrada(mensaje):
            while True:
                try:
                    valor = int(input(mensaje).strip())
                    if 0 <= valor <= 5:
                        return valor
                    else:
                        print("El valor debe ser un número entre 0 y 5.")
                except ValueError:
                    print("Entrada inválida. Por favor, ingrese un número entre 0 y 5.")
        
        # Lista para almacenar los índices de las filas a eliminar
        indices_a_eliminar = []
        
        # Iterar sobre todas las imágenes en el dataframe original
        for index in range(len(df_consolidado)):
            imagen = df_consolidado.iloc[index, 0]  # Suponiendo que 'Frame' es la primera columna
            print("Etiqueta original: ", df_consolidado.iloc[index, 1])
            print("Etiqueta homologada: ", df_consolidado.iloc[index, 2])
            
            if isinstance(imagen, np.ndarray):
                if imagen.ndim == 2:  # Si la imagen es en escala de grises
                    # Convertir a formato BGR para mostrarla en OpenCV
                    imagen_bgr = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
                elif imagen.ndim == 3 and imagen.shape[2] == 1:  # Si es una imagen en escala de grises con un canal
                    imagen_bgr = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
                elif imagen.ndim == 3 and imagen.shape[2] == 3:  # Si es una imagen en formato BGR
                    imagen_bgr = imagen.copy()  # No es necesario convertir
        
                imagen_bgr_original = imagen_bgr.copy()
                # Procesar la malla facial con mediapipe
                resultados = MallaFacial.process(imagen_bgr)
                
                # Verificar si se detectaron rostros antes de continuar
                if resultados.multi_face_landmarks:
                    # Extraer puntos de la malla facial
                    puntos_malla = extraer_puntos_malla(resultados)
                    
                    # Dibujar la malla facial sobre la imagen original
                    imagen_malla_sobre_negro = np.zeros_like(imagen_bgr, dtype=np.uint8)
                    for rostro in resultados.multi_face_landmarks:
                        mpDibujo.draw_landmarks(imagen_malla_sobre_negro, rostro, mpMallaFacial.FACEMESH_CONTOURS, ConfDibu, ConfDibu) 
                        mpDibujo.draw_landmarks(imagen_bgr, rostro, mpMallaFacial.FACEMESH_CONTOURS, ConfDibu, ConfDibu) 
                    
                    # Calcular la emoción utilizando DeepFace
                    resultado_emocion = DeepFace.analyze(imagen_bgr, actions=['emotion'], enforce_detection=False)
                    # Verificar si el resultado es una lista
                    if isinstance(resultado_emocion, list):
                        emocion_predicha = resultado_emocion[0]['dominant_emotion']
                    else:
                        emocion_predicha = resultado_emocion['dominant_emotion']
                    
                    # Calcular la emoción utilizando el modelo fer2013_mini_XCEPTION
                    preprocessed_image = preprocess_image(imagen_bgr)
                    predictions = model.predict(preprocessed_image)
                    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
                    emocion_predicha_xception = emotion_labels[np.argmax(predictions)]
                            
                    # Dibujar la malla facial sobre la imagen recortada y redimensionada
                    imagen_recortada_redimensionada = recortar_y_redimensionar(imagen_malla_sobre_negro, puntos_malla)
                    imagen_bgr_ = recortar_y_redimensionar(imagen_bgr, puntos_malla)
                                    
                    if imagen_recortada_redimensionada.shape[-1] == 3:
                        imagen_recortada_redimensionada_best_model = cv2.cvtColor(imagen_recortada_redimensionada, cv2.COLOR_BGR2GRAY)
                        
                    # Preparar la imagen para la predicción del modelo
                    frame_expanded = np.expand_dims(imagen_recortada_redimensionada_best_model, axis=-1)
                    frame_expanded = np.expand_dims(frame_expanded, axis=0)
                    frame_expanded = frame_expanded.astype(np.float32) / 255.0  # Normalización
                    
                    # Predecir la emoción
                    emotion_prediction = best_model.predict(frame_expanded)
                    # Obtener la emoción predicha
                    predicted_emotion_index = np.argmax(emotion_prediction) + 1
                    # Verificar si el índice predicho está en el mapeo de etiquetas
                    if predicted_emotion_index in mapeo_etiquetas:
                        predicted_emotion_label = mapeo_etiquetas[predicted_emotion_index]
                    else:
                        predicted_emotion_label = 'N/A'
                        print(f"Índice de emoción predicha fuera del rango: {predicted_emotion_index}")
                    
                    print("Emocion predicha DeepFace: ", emocion_predicha)
                    print(f'Emoción predicha fer2013_mini_XCEPTION: {emocion_predicha_xception}')
                    print("Emocion predicha mejor modelo: ", predicted_emotion_label)
                            
                    # Visualizar las imágenes
                    cv2.imshow('Imagen recortada y redimensionada con Malla Facial', imagen_bgr_)
                    cv2.imshow('Imagen original', imagen_bgr_original)
                    cv2.imshow('Imagen recortada y redimensionada con Malla Facial sobre fondo negro', imagen_recortada_redimensionada)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                     
                    # Preguntar si se quiere eliminar la imagen
                    delete = input("¿Deseas eliminar esta imagen? (Y/N): ").strip().lower()
                    clear_output(wait=True)  # Limpiar la salida para la siguiente imagen
                    
                    # Convertir la imagen a escala de grises
                    imagen_final_gris = cv2.cvtColor(imagen_recortada_redimensionada, cv2.COLOR_BGR2GRAY)
                
                    df_consolidado.at[index, 'Frame'] = imagen_final_gris
                    df_consolidado.at[index, 'Puntos_Malla'] = puntos_malla
                    
                    # Preguntar por el nuevo valor para la lista
                    if delete != 'y':
                        new_value_1 = validar_entrada("Ingrese el nuevo valor para el primer elemento de la lista (0-5): ")    
                        df_consolidado.at[index, 'Etiqueta_Manual'] = int(new_value_1)
                        print("Nuevas etiquetas: ",df_consolidado.iloc[index, 6])
                        print("Tamaño del nuevo frame: ",df_consolidado.iloc[index, 0].shape)
                        print("Numero de puntos en la malla: ",len(df_consolidado.iloc[index, 7]))
                    else:
                        indices_a_eliminar.append(index)
                        
                    df_consolidado.at[index, 'Etiqueta_DeepFace'] = emocion_predicha
                    df_consolidado.at[index, 'Etiqueta_Fer2013'] = emocion_predicha_xception
                    df_consolidado.at[index, 'Etiqueta_Modelo'] = predicted_emotion_label
                    print("--------------------------------------------------------------------------")
                else:
                    print("No se detectaron rostros en la imagen")
                    indices_a_eliminar.append(index)
            
            else:
                print("La imagen no es un numpy array")
                indices_a_eliminar.append(index)
        
        # Eliminar las filas que no sirven
        df_consolidado = df_consolidado.drop(indices_a_eliminar).reset_index(drop=True)
        
        df_consolidado['Frame'] = df_consolidado['Frame'].apply(lambda x: json.dumps(x.tolist()))
        df_consolidado['Etiqueta'] = df_consolidado['Etiqueta'].apply(lambda x: json.dumps(x))
        df_consolidado['Etiqueta_Homologada'] = df_consolidado['Etiqueta_Homologada'].apply(lambda x: json.dumps(x))
        df_consolidado['Puntos_Malla'] = df_consolidado['Puntos_Malla'].apply(lambda x: json.dumps(x))
        
        video_name = os.path.splitext(video)[0]
        
        df_consolidado.to_csv(f'D:/OneDrive - Pontificia Universidad Javeriana/Codigo/Datos_Manual_CNN/Consolidado_Video_{video_name}_Etiqueta_Manual.csv', index=False)
        print(f"Se guardo la base, Consolidado_Video_{video_name}_Etiqueta_Manual.csv")
        print("--------------------------------------------------------------------------")
#%%    
# Visualizar la imagen en la fila 12, columna 'Frame'
numero = 1

imagen = df_consolidado.iloc[numero, 0]  # Suponiendo que 'Frame' es la primera columna
print(df_consolidado.iloc[numero, 1])
print(df_consolidado.iloc[numero, 2])
print(imagen.shape)
# Verificar que la imagen es un numpy array antes de visualizarla
if isinstance(imagen, np.ndarray):
    # Convertir a tipo uint8 si no lo es
    if imagen.dtype != np.uint8:
        imagen = imagen.astype(np.uint8)
    # Visualizar la imagen
    cv2.imshow('Imagen', imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("La imagen no es un numpy array")
#%%
import pandas as pd
import numpy as np
import scipy.stats as stats

# Filtrar los registros que tienen valor > 0 en el tercer elemento de 'columna_homologada'
df_filtrado = df_consolidado[df_consolidado['Var_Respuestas'].apply(lambda x: x[1] > 0)]

# Extraer los datos de interés
antes = df_filtrado['Var_Respuestas'].apply(lambda x: x[1])
despues = df_filtrado['Var_Respuestas'].apply(lambda x: x[3])

#Asegurarse de que ambos son listas de enteros o floats
antes = np.array(antes, dtype=np.float64)
despues = np.array(despues, dtype=np.float64)

# Prueba t de Student para muestras relacionadas
#Hipótesis nula: La media de las diferencias es igual a cero.
#Hipótesis alternativa: La media de las diferencias no es igual a cero.
#Supuestos:
#Las diferencias son aproximadamente normales.
#Los pares de datos son relacionados.
#Si el p-valor es menor que el nivel de significancia (típicamente 0.05), rechazas la hipótesis nula y concluyes que las medias son significativamente diferentes.
#Si el p-valor es mayor que el nivel de significancia, no rechazas la hipótesis nula y concluyes que no hay suficiente evidencia para decir que las medias son diferentes.
t_stat, p_value = stats.ttest_rel(antes, despues)
print("Prueba t de Student para muestras relacionadas")
print('t-statistic:', t_stat)
print('p-value:', p_value)

# Interpretación
if p_value < 0.05:
    print("Rechazamos la hipótesis nula: Las medianas son significativamente diferentes.")
else:
    print("No rechazamos la hipótesis nula: No hay suficiente evidencia para concluir que las medianas son diferentes.")
#%%
# Prueba de Wilcoxon para muestras relacionadas
stat, p_value = stats.wilcoxon(antes, despues)
print("Prueba de Wilcoxon para muestras relacionadas")
print('Estadístico de prueba:', stat)
print('p-valor:', p_value)

# Interpretación
if p_value < 0.05:
    print("Rechazamos la hipótesis nula: Las medianas son significativamente diferentes.")
else:
    print("No rechazamos la hipótesis nula: No hay suficiente evidencia para concluir que las medianas son diferentes.")
    
# Calcular y comparar medianas
mediana_antes = np.median(antes)
mediana_despues = np.median(despues)

print('Mediana antes de escuchar la canción:', mediana_antes)
print('Mediana después de escuchar la canción:', mediana_despues)