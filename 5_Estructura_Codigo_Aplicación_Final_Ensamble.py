### Inicio de todo el proceso 
from tkinter import Tk, Label, Button, Radiobutton, IntVar
from PIL import Image, ImageTk
import cv2
import imutils
import mediapipe as mp
import math
import os
import random
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
import spotipy
import pandas as pd
from spotipy import SpotifyException
import psutil
import pyautogui
from time import sleep
import numpy as np
import datetime
import gc
import webbrowser as web
from tensorflow.keras.models import load_model
import pickle
from scipy.stats import mode  # Importar mode para calcular la moda
import time  # Importa time para manejar el temporizador

# Cargar modelos
best_model_VF_0 = load_model('Best_Emotion_Model_VF_0.h5', compile=False)
best_model_VF_0.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

best_model_VF_2 = load_model('Best_Emotion_Model_VF_2.h5', compile=False)
best_model_VF_2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Mostrar la estructura del mejor modelo
#best_model_VF.summary()

# Cargar datos de música
music_df = pd.read_csv("data_moods.csv")
music_df = music_df[['name','artist','mood','popularity']]
music_df["cancion"] = music_df["artist"] + " " + music_df["name"]

# Mapeo de etiquetas
mapeo_etiquetas = {
    1: 'Triste',
    2: 'Enojado',
    3: 'Neutral',
    4: 'Sorprendido',
    5: 'Feliz'
}

# Mapeo de valores numéricos a texto para feedback
feedback_mapping = {
    1: "like",
    2: "neutral",
    3: "dislike"
}

# credenciales spotify
client_id = '0fc0ba04f0e14cd1b04bf63dde526b62'
client_secret = '63ce7f4fc3474eb3baba0565daeebe5f'
# authenticate
sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id, client_secret))

# Función para integrar las predicciones de los dos modelos
def integrar_predicciones(frame_expanded):
    pred_VF_0 = best_model_VF_0.predict(frame_expanded)
    pred_VF_2 = best_model_VF_2.predict(frame_expanded)

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
    
def limpiar_variables_globales():
    variables_globales = list(globals().keys())
    variables_a_none = ['rad11', 'rad21', 'rad31', 'Info_11', 'btnSiguiente_1', 
                        'texto', 'texto_mus', 'cancion_sonar', 'cancion_sonar_', 'texto_mus_F']
    variables_a_cero = ['selected', 'selected_F']
    for variable in variables_globales:
        if variable in variables_a_none:
            globals()[variable] = None   
        elif variable in variables_a_cero:
            globals()[variable].set(0)
            
# Incluir nuevas canciones en rewards_table si no están presentes
def actualizar_rewards_table():
    for cancion in music_df['cancion']:
        if cancion not in rewards_table.index:
            rewards_table.loc[cancion] = [0, 0]
            
def reproducir_cancion(cancion): 
    result = sp.search(cancion)
    web.open(result["tracks"]["items"][0]["uri"])

# Definir la función para recomendar canciones basada en la emoción
def Recommend_Songs(pred_class):
    mood_mapping_1 = {2: 'Happy', 5: 'Happy', 3: 'Calm', 1: 'Sad', 4: 'Happy', '': 'Calm'}
    mood_mapping_2 = {2: 'Energetic', 5: 'Energetic', 3: 'Calm', 1: 'Calm', 4: 'Energetic', '': 'Calm'}
    mood_mapping = random.choice([mood_mapping_1, mood_mapping_2])
    mood = mood_mapping.get(pred_class, 'Happy')
    Play = music_df[music_df['mood'] == mood]
    return Play['cancion'].tolist()

# Guardar el modelo de recompensas (tabla de recompensas), epsilon, episode y registro de episodios
def guardar_modelo(ruta='Musica_modelo_recompensas.pkl', epsilon_ruta='Musica_epsilon_value.pkl', log_ruta='Musica_log_episodios.csv'):
    with open(ruta, 'wb') as archivo:
        pickle.dump(rewards_table, archivo)
    with open(epsilon_ruta, 'wb') as archivo:
        pickle.dump((epsilon, episode), archivo)  # Guardamos epsilon y episode juntos
    episodio_log.to_csv(log_ruta, index=False)
    print("Modelo de recompensas, epsilon, episodio y log de episodios guardados exitosamente.")

# Cargar el modelo de recompensas (tabla de recompensas), epsilon, episode y registro de episodios
def cargar_modelo(ruta='Musica_modelo_recompensas.pkl', epsilon_ruta='Musica_epsilon_value.pkl', log_ruta='Musica_log_episodios.csv'):
    global rewards_table, epsilon, episode, episodio_log
    try:
        with open(ruta, 'rb') as archivo:
            rewards_table = pickle.load(archivo)
        print("Modelo de recompensas cargado exitosamente.")
    except FileNotFoundError:
        print("No se encontró un modelo de recompensas previo. Se creará uno nuevo.")
        rewards_table = pd.DataFrame(0, index=music_df['cancion'], columns=['Total_Rewards', 'Count'])
    
    try:
        with open(epsilon_ruta, 'rb') as archivo:
            data = pickle.load(archivo)
            # Verifica si el archivo contiene solo epsilon o ambos valores
            if isinstance(data, tuple):
                epsilon, episode = data  # Desempaca ambos valores
            else:
                epsilon = data  # Solo contiene epsilon
                episode = 0  # Inicializa episode en 0
        print(f"Epsilon cargado exitosamente: {epsilon}, Episodio actual: {episode}")
    except FileNotFoundError:
        epsilon = 1.0
        episode = 0
        print("No se encontró un valor de epsilon ni episodio previo. Se inicia con epsilon=1.0 y episode=0.")
    
    try:
        episodio_log = pd.read_csv(log_ruta)
        print("Log de episodios cargado exitosamente.")
    except FileNotFoundError:
        episodio_log = pd.DataFrame(columns=['Episode', 'Recommended_Song', 'Initial_State', 'Reward', 'Final_State'])

# Función para calcular la recompensa con penalización leve y castigo mínimo
def calcular_recompensa(estado_inicial, estado_final):
    delta_emocional = estado_final - estado_inicial
    if delta_emocional > 0:
        return np.sqrt(delta_emocional)  # Crecimiento suavizado para premios
    elif delta_emocional == 0:
        if estado_final == 5:
            return 0.2  # Recompensa leve si estado_final es 5
        else:
            return -0.2  # Penalización leve si estado_final no es 5
    else:
        return max(-4, -1 * np.sqrt(abs(delta_emocional)))  # Castigo limitado a -4

# Función para actualizar la recompensa en base a la retroalimentación explícita del usuario
def actualizar_recompensa_por_feedback(cancion, feedback):
    if feedback == "like":
        rewards_table.loc[cancion, 'Total_Rewards'] += 1  # Aumentar recompensa
    elif feedback == "neutral":
        rewards_table.loc[cancion, 'Total_Rewards'] -= 0.2  # Penalización leve
    elif feedback == "dislike":
        rewards_table.loc[cancion, 'Total_Rewards'] -= 1  # Reducir recompensa
    rewards_table.loc[cancion, 'Count'] += 1  # Incrementar el conteo de interacciones
    
# Función para elegir una canción usando epsilon-greedy
def elegir_cancion(canciones_filtradas):
    if np.random.rand() < epsilon:
        return random.choice(canciones_filtradas)
    else:
        filtered_rewards = rewards_table.loc[canciones_filtradas]
        avg_rewards = filtered_rewards['Total_Rewards'] / (filtered_rewards['Count'] + 1e-8)
        return avg_rewards.idxmax()   

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

# Definiciones generales 
rad11 = None
rad21 = None
rad31 = None
Info_11 = None
btnSiguiente_1 = None

# Inicializar y cargar rewards_table, epsilon y log de episodios
cargar_modelo()
actualizar_rewards_table()
    
# Parámetros de epsilon-greedy
epsilon_min = 0.1
epsilon_decay = 0.995

#Estructura
def iniciar():
    global cap, out, texto, texto_mus, texto_mus_F, cancion_sonar_, etiquetas_recientes, estado_inicial, tiempo_inicial, estado_final, cancion_sonar, recompensa_calculada, tiempo_final
        
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # Inicializar el objeto de escritura de video
    nombre_archivo = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S") + '_video.avi'
    out = cv2.VideoWriter(nombre_archivo, cv2.VideoWriter_fourcc(*'XVID'), 10, (640, 480))
    
    estado_inicial = None
    estado_final = None
    tiempo_inicial = None
    tiempo_final = None
    # Variables globales para el estado inicial y el estado final
    cancion_sonar = None 
    # Añade la variable global para el control de ejecución
    recompensa_calculada = False
    etiquetas_recientes = []  # Reinicia las etiquetas recientes
    visualizar()
    
def visualizar():
    global cap, out, cancion_sonar, texto, texto_mus, texto_mus_F, cancion_sonar_, etiquetas_recientes, estado_inicial, tiempo_inicial, estado_final, episode, epsilon, episodio_log, recompensa_calculada, tiempo_final
    if cap is not None:    
        ret, frame = cap.read()
        if ret == True:
            frame = imutils.resize(frame, width=640) 
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resultados = MallaFacial.process(frameRGB)
            
            # Extraer puntos de la malla facial
            puntos_malla = extraer_puntos_malla(resultados)
            
            # Inicializar variables para las imágenes a mostrar
            imagen_recortada_redimensionada = None
            
            if resultados.multi_face_landmarks is not None:
                for idx, rostros in enumerate(resultados.multi_face_landmarks):
                    
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
                    
                    # Integrar predicciones de ambos modelos
                    predicted_emotion_index_VF, pred_emocion = integrar_predicciones(frame_expanded)
                    etiquetas_recientes.append(predicted_emotion_index_VF)

                    if len(etiquetas_recientes) > 30:  ### Para estabilizar la etiqueta se calcula la moda de las ultimas 20 respuestas del modelo
                        etiquetas_recientes.pop(0)
                        
                    # Calcular la moda de etiquetas_recientes si tiene longitud 20
                    if len(etiquetas_recientes) == 30 and estado_inicial is None:
                        
                        # Obtener los últimos 20 elementos de etiquetas_recientes
                        ultimas_20_etiquetas = etiquetas_recientes[-20:]
                        moda_result = mode(ultimas_20_etiquetas)
                        
                        # Obtén el valor de la moda
                        if isinstance(moda_result.mode, np.ndarray):
                            estado_inicial = moda_result.mode[0]  # Si es un array, toma el primer elemento
                            estado_inicial = mapeo_etiquetas[estado_inicial]
                        else:
                            estado_inicial = moda_result.mode  # Si es un solo valor, úsalo directamente
                            estado_inicial = mapeo_etiquetas[estado_inicial]
                        tiempo_inicial = time.time()  # Guarda el tiempo del estado inicial
                        print("Estado inicial:", estado_inicial)

                    # Calcular el estado final después de 30 segundos
                    if tiempo_inicial and (time.time() - tiempo_inicial) >= 20 and estado_final is None:
                        # Obtener los últimos 20 elementos de etiquetas_recientes
                        ultimas_20_etiquetas = etiquetas_recientes[-20:]
                        moda_result_final = mode(ultimas_20_etiquetas)
                        
                        # Obtén el valor de la moda
                        if isinstance(moda_result_final.mode, np.ndarray):
                            estado_final = moda_result_final.mode[0]  # Si es un array, toma el primer elemento
                            estado_final = mapeo_etiquetas[estado_final]
                        else:
                            estado_final = moda_result_final.mode  # Si es un solo valor, úsalo directamente
                            estado_final = mapeo_etiquetas[estado_final]
                            
                        print("Estado final:", estado_final)

                    # Verificar si el índice predicho está en el mapeo de etiquetas
                    if predicted_emotion_index_VF in mapeo_etiquetas:
                        predicted_emotion_index_VF = mapeo_etiquetas[predicted_emotion_index_VF]
                    else:
                        predicted_emotion_index_VF = 'N/A'
                        print(f"Índice de emoción predicha fuera del rango: {predicted_emotion_index_VF}")
 
                    texto = predicted_emotion_index_VF
                    # Superponer el texto en el video
                    if texto:
                        fondo = np.ones((50, len(texto) * 20, 3), dtype=np.uint8) # Fondo
                        cv2.putText(fondo, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)  # Agregar texto 
                        frame[:50, :len(texto) * 20] = fondo  # Superponer el texto en el cuadro de vídeo                        
                     
                    if estado_inicial:
                        fondo = np.ones((50, len(estado_inicial) * 20, 3), dtype=np.uint8) # Fondo
                        cv2.putText(fondo, estado_inicial, (15 + fondo.shape[1] - len(estado_inicial) * 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)  # Agregar texto 
                        frame[:50, -len(estado_inicial) * 20:] = fondo   # Superponer el texto en el cuadro de vídeo
        
        
                    if estado_final:
                        longitud_texto = len(estado_final) * 20
                        altura_fondo = 50
                        fondo = np.ones((altura_fondo, longitud_texto, 3), dtype=np.uint8)  # Fondo
                        cv2.putText(fondo, estado_final, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  
                        frame[-altura_fondo - 20:-20, -longitud_texto:] = fondo 
                                            

                    if cancion_sonar:
                        cancion_sonar_ = str(cancion_sonar)
                        max_text_length = min(len(cancion_sonar_) * 10, frame.shape[1])  # Ajusta el texto para que no exceda el ancho de frame
                        altura = 25  # Mitad de 50
                        fondo_ = np.ones((altura, max_text_length, 3), dtype=np.uint8)  # Fondo ajustado
                        # Escribe el texto en el fondo
                        cv2.putText(fondo_, cancion_sonar_[:max_text_length // 10], (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        # Superponer el fondo en frame
                        frame[-altura:, :max_text_length] = fondo_
                        
                    # Comprobamos si estado_inicial ya tiene un valor válido y ejecutamos crear_botones_musica si es así
                    if estado_inicial and not hasattr(root, 'musica_creada'):
                        root.musica_creada = True  # Marcador para evitar múltiples ejecuciones
                        crear_botones_musica()

                    # Condición para ejecutar la función una sola vez
                    if estado_final and selected_F.get() != 0 and not recompensa_calculada:
                        
                        # Convertir el estado emocional de texto a un valor numérico
                        estado_inicial_num = list(mapeo_etiquetas.keys())[list(mapeo_etiquetas.values()).index(estado_inicial)]
                        estado_final_num = list(mapeo_etiquetas.keys())[list(mapeo_etiquetas.values()).index(estado_final)]
                        reward = calcular_recompensa(estado_inicial_num, estado_final_num)
                        # Actualizar tabla de recompensas para la canción recomendada
                        rewards_table.loc[cancion_sonar, 'Total_Rewards'] += reward
                        rewards_table.loc[cancion_sonar, 'Count'] += 1
                        
                        feedback = feedback_mapping.get(selected_F.get(), "neutral")  # Si no se encuentra, devuelve "Unknown"
                        print("recomendación de la persona",feedback)
                        actualizar_recompensa_por_feedback(cancion_sonar, feedback)
                        print("rewards ", rewards_table.loc[cancion_sonar, 'Total_Rewards'])
                        # Agregar registro de episodio al log
                        episodio_log = pd.concat([episodio_log, pd.DataFrame({
                            'Episode': [episode + 1],
                            'Initial_State': [estado_inicial],
                            'Recommended_Song': [cancion_sonar],
                            'Final_State': [estado_final],
                            'Reward': [rewards_table.loc[cancion_sonar, 'Total_Rewards']],
                            'Feedback': [feedback]
                        })], ignore_index=True)
                        
                        # Normalizar recompensas cada 15 episodios
                        if (episode + 1) % 15 == 0:
                            rewards_table['Total_Rewards'] = rewards_table['Total_Rewards'] / (rewards_table['Count'] + 1e-8)
                            rewards_table['Count'] = rewards_table['Count'] / (rewards_table['Count'] + 1e-8)
                    
                        # Actualizar epsilon para reducir exploración y reiniciar a 0.5 cada 250 episodios
                        epsilon = max(epsilon_min, epsilon * epsilon_decay)
                        if (episode + 1) % 250 == 0:
                            epsilon = 0.5
                    
                        print(f"Episode {episode + 1}: Recommended '{cancion_sonar}' ---- emoción inicial: {estado_inicial}, ---- Reward: {reward}, --- emoción final: {estado_final}, --- Feedback: {feedback}")
                        # Marcar la recompensa como calculada para que no se vuelva a ejecutar
                        recompensa_calculada = True
                        tiempo_final = time.time()  # Guarda el tiempo del estado inicial
                        
                    if tiempo_final and (time.time() - tiempo_final) >= 5 :
                        finalizar()
                
            out.write(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=im)
            ### Para ver el video de la persona desmarcar el siguiente comando
            lblVideo.configure(image=img) # img, imagen_recortada_redimensionada
            lblVideo.image = img
            lblVideo.after(5, visualizar) 
            
            # En la función visualizar(), antes de intentar mostrar la imagen recortada
            if imagen_recortada_redimensionada is not None:
                im_recortada = Image.fromarray(imagen_recortada_redimensionada)
                img_recortada = ImageTk.PhotoImage(image=im_recortada)
                # Para ver la malla de la persona
                lblRecortada.configure(image=img_recortada)
                lblRecortada.image = img_recortada
            else:
                print("No se detectaron puntos de malla facial; imagen recortada es None.")
            
        else:
            lblVideo.image = ""
            lblRecortada.image = ""
            cap.release()
            out.release()

def finalizar():
    global cap, out, episode, recompensa_calculada
    cap.release()
    out.release()
    #sp.pause_playback() 
    os.system("taskkill /f /im spotify.exe") 
    destruir_botones_musica()
    # Llamar a la función para limpiar las variables globales
    limpiar_variables_globales()
    # Guardar tabla de recompensas, log de episodios, y valor de epsilon al finalizar
    # Guardar `episode` junto con el modelo
    episode += 1
    guardar_modelo()
    delattr(root, 'musica_creada')
    recompensa_calculada = False
        
def crear_botones_musica(): 
    global cancion_sonar, estado_inicial, estado_final
    musica = Recommend_Songs(estado_inicial)  
    # Elegir canción con epsilon-greedy
    cancion_sonar = elegir_cancion(musica)
    print(f"La canción escogida: {cancion_sonar}, para la emoción: {estado_inicial}")
    reproducir_cancion(cancion_sonar)
    
    global rad11, rad21, rad31, Info_11, btnSiguiente_1
    rad11 = Radiobutton(root, text="Like", width=20, value=1, variable=selected_F, font=("Helvetica", 16))  
    rad21 = Radiobutton(root, text="Neutral", width=20, value=2, variable=selected_F, font=("Helvetica", 16))
    rad31 = Radiobutton(root, text="Dislike", width=20, value=3, variable=selected_F, font=("Helvetica", 16))
    rad11.grid(column=2, row=1)
    rad21.grid(column=2, row=2)
    rad31.grid(column=2, row=3)
    Info_11 = Label(root, text= "¿Te ha gustado la canción?", bg="green", fg="white", font=("Helvetica", 15))
    Info_11.grid(column=2, row=0, padx=(5, 5), pady=5, sticky="we") 
        
def destruir_botones_musica():
    global rad11, rad21, rad31, Info_11, btnSiguiente_1
    for rad in [rad11, rad21, rad31, Info_11, btnSiguiente_1]:
        if rad is not None:
            rad.destroy()
          
cap = None
root = Tk()
selected = IntVar()
selected_F = IntVar()
btnIniciar = Button(root, text="Iniciar", width=20, command=iniciar, font=("Helvetica", 12))
btnIniciar.grid(column=0, row=0, padx=5, pady=5, columnspan=2)

lblVideo = Label(root)
lblVideo.grid(column=0, row=1, columnspan=1, rowspan=7)
lblRecortada = Label(root)  # Nueva etiqueta para la imagen recortada
lblRecortada.grid(column=1, row=1, columnspan=1, rowspan=7)

# Inicializar Mediapipe para la malla facial
mpMallaFacial = mp.solutions.face_mesh
MallaFacial = mpMallaFacial.FaceMesh(max_num_faces=1)
mpDibujo = mp.solutions.drawing_utils
ConfDibu = mpDibujo.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=0)

root.mainloop()