import os
import tensorflow as tf
from tensorflow.python.client import device_lib
import pandas as pd
import numpy as np
import json
import cv2
from tensorflow.keras import layers, models
from skimage.transform import resize
import random
import h5py
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Definir el número de segmentos para train y test
num_total_segments = 20

# Lista para almacenar los DataFrames cargados de train
total_dfs = []

# Cargar cada segmento de train y agregarlo a la lista
for i in range(num_total_segments):
    segment_df = pd.read_csv(f'D:/OneDrive - Pontificia Universidad Javeriana/Codigo/0_Datos_Modelo_RL/Consolidado_Malla_Facial_Total_20240718_segment_{i}.csv')
    
    print(segment_df.shape[0])
    
    # Seleccionar la columna 'Etiqueta_Homologada' y quedarse con los registros únicos
    segment_df_ = segment_df[['Etiqueta_Homologada']].drop_duplicates().reset_index(drop=True)

    print(segment_df_.shape[0])
    
    total_dfs.append(segment_df_)
    print(f'Segmento total' , str(i), 'cargado.')
    print("-------------------------------------------------------------")

# Concatenar los DataFrames de train en uno solo
total_df_consolidado = pd.concat(total_dfs, ignore_index=True)
print("Tamaño del conjunto de total consolidado:", total_df_consolidado.shape)

# Deserializar las columnas y agregar la dimensión del canal a las imágenes
total_df_consolidado['Etiqueta_Homologada'] = total_df_consolidado['Etiqueta_Homologada'].apply(lambda x: json.loads(x)) 
print("ya quedo la base total")

# Crear la nueva columna 'estado_inicial' usando el segundo valor en 'Var_Respuestas'
total_df_consolidado['estado_inicial'] = total_df_consolidado['Etiqueta_Homologada'].apply(lambda x: x[1])
total_df_consolidado['estado_final'] = total_df_consolidado['Etiqueta_Homologada'].apply(lambda x: x[3])
total_df_consolidado['canción_escuchada'] = total_df_consolidado['Etiqueta_Homologada'].apply(lambda x: x[2])

print(total_df_consolidado.shape[0])
total_df_consolidado = total_df_consolidado[['estado_inicial', 'canción_escuchada', 'estado_final']].drop_duplicates().reset_index(drop=True)
print(total_df_consolidado.shape[0])

# Filtrar los registros donde 'estado_final' no es igual a cero
total_df_consolidado = total_df_consolidado[total_df_consolidado['estado_inicial'] != 0].reset_index(drop=True)
print(total_df_consolidado.shape[0])
total_df_consolidado = total_df_consolidado[total_df_consolidado['estado_final'] != 0].reset_index(drop=True)
print(total_df_consolidado.shape[0])
total_df_consolidado = total_df_consolidado[total_df_consolidado['canción_escuchada'] != 0].reset_index(drop=True)
print(total_df_consolidado.shape[0])

# Resumen del número de datos por etiqueta
Etiqueta_Manual_summary = total_df_consolidado['estado_inicial'].value_counts()
print("estado_inicial: ",Etiqueta_Manual_summary)

# Resumen del número de datos por etiqueta
Etiqueta_Manual_summary = total_df_consolidado['estado_final'].value_counts()
print("estado_final: ",Etiqueta_Manual_summary)

#%%
# Supongamos que tienes dos listas de estados originales y estados transicionados
original_states = total_df_consolidado['estado_inicial']
new_states = total_df_consolidado['estado_final']

#Asegurarse de que ambos son listas de enteros o floats
original_states = np.array(original_states, dtype=np.float64)
new_states = np.array(new_states, dtype=np.float64)

unique_states = [1,2,3,4,5]
#unique_states = ['Enojado', 'Feliz', 'Neutral', 'Triste', 'Sorprendido', 'No_Disponible']
    
# Crear un DataFrame de pandas para la matriz de transición
transition_matrix = pd.DataFrame(0, index=unique_states, columns=unique_states)

# Rellenar la matriz de transición
for orig, new in zip(original_states, new_states):
    transition_matrix.loc[orig, new] += 1

print("Matriz de Transición (Frecuencia):")
print(transition_matrix)

# Convertir la matriz de frecuencia a una matriz de probabilidades (opcional)
transition_matrix_prob = transition_matrix.div(transition_matrix.sum(axis=1), axis=0)

print("\nMatriz de Transición (Probabilidad):")
print(transition_matrix_prob)
#%%
import numpy as np
import pandas as pd
import pickle
import random

# Cargar datos de música
music_df = pd.read_csv("data_moods.csv")
music_df["cancion"] = music_df["artist"] + " " + music_df["name"]

# Incluir nuevas canciones en rewards_table si no están presentes
def actualizar_rewards_table():
    for cancion in music_df['cancion']:
        if cancion not in rewards_table.index:
            rewards_table.loc[cancion] = [0, 0]

# Definir la función para recomendar canciones basada en la emoción
def Recommend_Songs(pred_class):
    mood_mapping_1 = {2: 'Happy', 5: 'Happy', 3: 'Calm', 1: 'Sad', 4: 'Happy', '': 'Calm'}
    mood_mapping_2 = {2: 'Energetic', 5: 'Energetic', 3: 'Calm', 1: 'Calm', 4: 'Energetic', '': 'Calm'}
    mood_mapping = random.choice([mood_mapping_1, mood_mapping_2])
    mood = mood_mapping.get(pred_class, 'Happy')
    Play = music_df[music_df['mood'] == mood]
    return Play['cancion'].tolist()

# Guardar el modelo de recompensas (tabla de recompensas), epsilon y registro de episodios
def guardar_modelo(ruta='Musica_modelo_recompensas.pkl', epsilon_ruta='Musica_epsilon_value.pkl', log_ruta='Musica_log_episodios.csv'):
    with open(ruta, 'wb') as archivo:
        pickle.dump(rewards_table, archivo)
    with open(epsilon_ruta, 'wb') as archivo:
        pickle.dump(epsilon, archivo)
    episodio_log.to_csv(log_ruta, index=False)
    print("Modelo de recompensas, epsilon y log de episodios guardados exitosamente.")

# Cargar el modelo de recompensas (tabla de recompensas), epsilon y registro de episodios
def cargar_modelo(ruta='Musica_modelo_recompensas.pkl', epsilon_ruta='Musica_epsilon_value.pkl', log_ruta='Musica_log_episodios.csv'):
    global rewards_table, epsilon, episodio_log
    try:
        with open(ruta, 'rb') as archivo:
            rewards_table = pickle.load(archivo)
        print("Modelo de recompensas cargado exitosamente.")
    except FileNotFoundError:
        print("No se encontró un modelo de recompensas previo. Se creará uno nuevo.")
        rewards_table = pd.DataFrame(0, index=music_df['cancion'], columns=['Total_Rewards', 'Count'])
    
    try:
        with open(epsilon_ruta, 'rb') as archivo:
            epsilon = pickle.load(archivo)
        print(f"Epsilon cargado exitosamente: {epsilon}")
    except FileNotFoundError:
        epsilon = 1.0
        print("No se encontró un valor de epsilon previo. Se inicia con epsilon=1.0.")
    
    try:
        episodio_log = pd.read_csv(log_ruta)
        print("Log de episodios cargado exitosamente.")
    except FileNotFoundError:
        episodio_log = pd.DataFrame(columns=['Episode', 'Recommended_Song', 'Initial_State', 'Reward', 'Final_State'])

# Inicializar y cargar rewards_table, epsilon y log de episodios
cargar_modelo()
actualizar_rewards_table()

# Parámetros de epsilon-greedy
epsilon_min = 0.1
epsilon_decay = 0.995

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
    
# Pre-entrenamiento con los datos reales
for index, row in total_df_consolidado.iterrows():
    estado_inicial = row['estado_inicial']
    estado_final = row['estado_final']
    cancion_escuchada = row['canción_escuchada']

    # Calcular recompensa y actualizar la tabla
    reward = calcular_recompensa(estado_inicial, estado_final)
    if cancion_escuchada in rewards_table.index:
        rewards_table.loc[cancion_escuchada, 'Total_Rewards'] += reward
        rewards_table.loc[cancion_escuchada, 'Count'] += 1
        
print("Pre-entrenamiento completo con datos reales.")

# Función para elegir una canción usando epsilon-greedy
def elegir_cancion(canciones_filtradas):
    if np.random.rand() < epsilon:
        return random.choice(canciones_filtradas)
    else:
        filtered_rewards = rewards_table.loc[canciones_filtradas]
        avg_rewards = filtered_rewards['Total_Rewards'] / (filtered_rewards['Count'] + 1e-8)
        return avg_rewards.idxmax()
#%%
# Simulación de recomendaciones incrementales
for episode in range(1):
    estado_inicial = random.choice(total_df_consolidado['estado_inicial'])
    canciones_filtradas = Recommend_Songs(estado_inicial)

    # Elegir canción con epsilon-greedy
    cancion_recomendada = elegir_cancion(canciones_filtradas)
    
    # Simulación del cambio de emoción después de escuchar la canción
    estado_final = random.choice(total_df_consolidado['estado_final'])

    # Calcular recompensa
    reward = calcular_recompensa(estado_inicial, estado_final)
    
    # Actualizar tabla de recompensas para la canción recomendada
    rewards_table.loc[cancion_recomendada, 'Total_Rewards'] += reward
    rewards_table.loc[cancion_recomendada, 'Count'] += 1

    # Simulación de retroalimentación del usuario
    feedback = random.choice(["like", "neutral", "dislike"])  # En producción, esto sería input del usuario
    actualizar_recompensa_por_feedback(cancion_recomendada, feedback)

    # Agregar registro de episodio al log
    episodio_log = pd.concat([episodio_log, pd.DataFrame({
        'Episode': [episode + 1],
        'Initial_State': [estado_inicial],
        'Recommended_Song': [cancion_recomendada],
        'Final_State': [estado_final],
        'Reward': [reward],
        'Feedback': [feedback]
    })], ignore_index=True)

    # Normalizar recompensas cada 15 episodios
    if (episode + 1) % 15 == 0:
        # Calcular el promedio y reemplazar en Total_Rewards
        rewards_table['Total_Rewards'] = rewards_table['Total_Rewards'] / (rewards_table['Count'] + 1e-8)
        # Reiniciar Count
        rewards_table['Count'] =  rewards_table['Count'] / (rewards_table['Count'] + 1e-8)

    # Guardar la tabla de recompensas, epsilon y log después de cada episodio
    guardar_modelo()

    # Actualizar epsilon para reducir exploración y reiniciar a 0.5 cada 250 episodios
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    if (episode + 1) % 250 == 0:
        epsilon = 0.5

    print(f"Episode {episode + 1}: Recommended '{cancion_recomendada}' ---- emoción inicial {estado_inicial}, ---- Reward: {reward}, --- emoción final: {estado_final}, --- Feedback: {feedback}")

print("Aprendizaje incremental completado.")