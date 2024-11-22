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

nombre_estudiante = "Alan_Brito"

def limpiar_variables_globales():
    
    # Obtener todas las variables globales
    variables_globales = list(globals().keys())    
    # Variables a establecer en None
    variables_a_none = ['rad1', 'rad2', 'rad3', 'rad4', 'rad5', 'Info_1', 'btnSiguiente', 'rad11', 'rad21', 'rad31', 'rad41', 'rad51', 'Info_11', 'btnSiguiente_1', 
                        'texto', 'texto_mus', 'cancion_sonar', 'cancion_sonar_', 'texto_mus_F'] 
    # Variables a establecer en cero
    variables_a_cero = ['selected', 'selected_F']
    # Establecer todas las variables globales a None o cero según corresponda
    for variable in variables_globales:
        if variable in variables_a_none:
            globals()[variable] = None   
        elif variable in variables_a_cero:
            globals()[variable].set(0)
        
music_df =pd.read_csv("data_moods.csv")
music_df = music_df[['name','artist','mood','popularity']]
Estados_animo =['Enojado','Feliz','Neutral','Triste','Sorprendido']

# credenciales spotify
client_id = '0fc0ba04f0e14cd1b04bf63dde526b62'
client_secret = '63ce7f4fc3474eb3baba0565daeebe5f'
# authenticate
sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id, client_secret))

def reproducir_cancion(cancion): 
    result = sp.search(cancion)
    web.open(result["tracks"]["items"][0]["uri"])

def Recommend_Songs(pred_class):
    mood_mapping_1 = {'Enojado': 'Happy',
        'Feliz': 'Happy',
        'Neutral': 'Calm',
        'Triste': 'Sad',
        'Sorprendido': 'Happy',
        '': 'Calm'}
    
    mood_mapping_2 = {'Enojado': 'Energetic',
        'Feliz': 'Energetic',
        'Neutral': 'Calm',
        'Triste': 'Calm',
        'Sorprendido': 'Energetic',
        '': 'Calm'}
    mood_mapping = random.choice([mood_mapping_1, mood_mapping_2])
    mood = mood_mapping.get(pred_class, 'Happy')  # Predeterminado a 'Happy' si no coincide
    Play = music_df[music_df['mood'] == mood]
    return Play

def segunda_derivada(puntos):
    if len(puntos) != 3:
        raise ValueError("Se necesitan exactamente tres puntos para calcular la segunda derivada.")
    y1, y2, y3 = [p[1] for p in puntos]
    segunda_derivada_aproximada = (y1 - 2 * y2 + y3)
    return segunda_derivada_aproximada

# Definiciones generales 
rad1 = None
rad2 = None
rad3 = None
rad4 = None
rad5 = None
Info_1 = None
btnSiguiente = None
rad11 = None
rad21 = None
rad31 = None
rad41 = None
rad51 = None
Info_11 = None
btnSiguiente_1 = None

#Estructura
def iniciar():
    global cap, out, texto, texto_mus, texto_mus_F, cancion_sonar_ 
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # Obtener la fecha y hora actual en formato YYYY-MM-DD_HH-MM-SS
    fecha_hora_actual = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    # Usar la fecha y hora actual en el nombre del archivo de salida
    out = cv2.VideoWriter(fecha_hora_actual + '_Estudiante_' + nombre_estudiante + '.avi', cv2.VideoWriter_fourcc(*'XVID'), 10, (640, 480))
    visualizar()
    crear_botones_imagen()
    
def visualizar():
    global cap, out, cancion_sonar, texto, texto_mus, texto_mus_F, cancion_sonar_    
    if cap is not None:    
        ret, frame = cap.read()
        if ret == True:
            frame = imutils.resize(frame, width=640) 
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resultados = MallaFacial.process(frameRGB)
            # Procesar el frame para el reconocimiento facial
            al, an, c = frame.shape
            px = []
            py = []
            lista = []
            if resultados.multi_face_landmarks is not None:
                for idx, rostros in enumerate(resultados.multi_face_landmarks):
                    #mpDibujo.draw_landmarks(frame, rostros, mpMallaFacial.FACEMESH_CONTOURS, ConfDibu, ConfDibu)
     
                    for id in range(468):
                        x, y = int(rostros.landmark[id].x * an), int(rostros.landmark[id].y * al)
                        px.append(x)
                        py.append(y)
                        lista.append([id, x, y])
                        
                        if len(lista) == 468:
                            # Ceja derecha
                            id1, x1, y1 = lista[67]
                            id2, x2, y2 = lista[65]
                            id3, x3, y3 = lista[158]
                            longitud1 = math.hypot(x3 - x2, y3 - y2)
                            longitud2 = math.hypot(x2 - x1, y2 - y1)
                            
                            # Ceja izquierda
                            id4, x4, y4 = lista[297]
                            id5, x5, y5 = lista[295]
                            id6, x6, y6 = lista[385]
                            longitud3 = math.hypot(x6 - x5, y6 - y5)
                            longitud4 = math.hypot(x5 - x4, y5 - y4)
                                
                            # Boca apertura
                            id7, x7, y7 = lista[1]
                            id8, x8, y8 = lista[0]
                            id9 ,x9, y9 = lista[17]
                            longitud5 = math.hypot(x9 - x8, y9 - y8)
                            longitud6 = math.hypot(x9 - x7, y9 - y7)

                            # Boca extremos
                            id10, x10, y10 = lista[61]
                            id11, x11, y11 = lista[14]
                            id12, x12, y12 = lista[291]
                            puntos = [(x10, y10), (x11, y11), (x12, y12)]
                            derivada = segunda_derivada(puntos)
                            
                            # Definir el texto y la posición
                            if longitud1/(longitud1+longitud2) > 0.375 and longitud5/longitud6 > 0.60:
                                texto = "Sorprendido"
                            elif 0.27 < longitud1/(longitud1+longitud2) < 0.375 and 0.38 < longitud5/longitud6 < 0.48 and -10 < derivada < 5:
                                texto = "Neutral"
                            elif 0.28 < longitud1/(longitud1+longitud2) < 0.375 and 0.48 < longitud5/longitud6 < 0.67 and derivada < -5:
                                texto = "Feliz"
                            elif 0.28 < longitud1/(longitud1+longitud2) < 0.39 and longitud5/longitud6 < 0.48 and derivada > 5:
                                texto = "Triste"
                            elif longitud1/(longitud1+longitud2) < 0.275 and 0.38 < longitud5/longitud6 < 0.48 and -5 < derivada < 5:
                                texto = "Enojado"
                            else:
                                texto = ""
                            
                            # Superponer el texto en el video
                            if texto:
                                fondo = np.ones((50, len(texto) * 20, 3), dtype=np.uint8) # Fondo
                                cv2.putText(fondo, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # Agregar texto 
                                frame[:50, :len(texto) * 20] = fondo  # Superponer el texto en el cuadro de vídeo
                            
                            if selected.get() == 0:
                                None
                            else: 
                                texto_mus = Estados_animo[selected.get()-1]
                                fondo = np.ones((50, len(texto_mus) * 20, 3), dtype=np.uint8) # Fondo
                                cv2.putText(fondo, texto_mus, (15 + fondo.shape[1] - len(texto_mus) * 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  
                                frame[:50, -len(texto_mus) * 20:] = fondo  

                            try:
                                if selected_F.get() > 0:
                                    texto_mus_F = Estados_animo[selected_F.get()-1]
                                    longitud_texto = len(texto_mus_F) * 20
                                    altura_fondo = 50
                                    fondo = np.ones((altura_fondo, longitud_texto, 3), dtype=np.uint8)  # Fondo
                                    cv2.putText(fondo, texto_mus_F, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  
                                    frame[-altura_fondo - 20:-20, -longitud_texto:] = fondo 
                                else: 
                                    None 
                            except:
                                pass
                            
                            try:
                                if cancion_sonar and cancion_sonar[0] is not None:
                                    cancion_sonar_ = str(cancion_sonar[0])
                                    max_text_length = len(cancion_sonar_) * 10  # Ajustar según el texto más largo
                                    altura = 25  # Mitad de 50
                                    fondo_ = np.ones((altura, max_text_length, 3), dtype=np.uint8) # Fondo
                                    cv2.putText(fondo_, cancion_sonar_, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  
                                    frame[-altura:, :max_text_length] = fondo_  
                                else:
                                    None  
                            except:
                                pass
            out.write(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=im)
            ### Para ver el video de la persona desmarcar el siguiente comando
            lblVideo.configure(image=img)
            lblVideo.image = img
            lblVideo.after(10, visualizar) 
        else:
            lblVideo.image = ""
            cap.release()
            out.release()

def finalizar():
    global cap, out
    cap.release()
    out.release()
    #sp.pause_playback() 
    os.system("taskkill /f /im spotify.exe") 
    destruir_botones_imagen()
    destruir_botones_musica()
    # Llamar a la función para limpiar las variables globales
    limpiar_variables_globales()

def crear_botones_imagen():
    global rad1, rad2, rad3, rad4, rad5, Info_1, btnSiguiente
    rad1 = Radiobutton(root, text="Enojado", width=20, value=1, variable=selected, font=("Helvetica", 16))  
    rad2 = Radiobutton(root, text="Feliz", width=20, value=2, variable=selected, font=("Helvetica", 16))
    rad3 = Radiobutton(root, text="Neutral", width=20, value=3, variable=selected, font=("Helvetica", 16))
    rad4 = Radiobutton(root, text="Triste", width=20, value=4, variable=selected, font=("Helvetica", 16))
    rad5 = Radiobutton(root, text="Sorprendido", width=20, value=5, variable=selected, font=("Helvetica", 16))
    rad1.grid(column=2, row=1)
    rad2.grid(column=2, row=2)
    rad3.grid(column=2, row=3)
    rad4.grid(column=2, row=4)
    rad5.grid(column=2, row=5)
    Info_1 = Label(root, text= "¿Que estado de animo tienes?", bg="green", fg="white", font=("Helvetica", 15))
    Info_1.grid(column=2, row=0, padx=(5, 5), pady=5, sticky="we") 
    btnSiguiente = Button(root, text="Siguiente", width=20, command=crear_botones_musica, font=("Helvetica", 12))
    btnSiguiente.grid(column=2, row=7, padx=5, pady=5)
    
def crear_botones_musica(): 
    global cancion_sonar
    texto_mus = Estados_animo[selected.get()-1] 
    musica = Recommend_Songs(texto_mus)
    cancion = musica["artist"] + " " + musica["name"]
    # Convertir la cadena a una lista de caracteres
    cancion_lista = list(cancion) 
    # Mezclar la lista de caracteres
    random.shuffle(cancion_lista)
    cancion_sonar = cancion_lista[:1]
    print(str(cancion_sonar[0]))    
    reproducir_cancion(cancion_sonar)
    destruir_botones_imagen()    
    global rad11, rad21, rad31, rad41, rad51, Info_11, btnSiguiente_1
    rad11 = Radiobutton(root, text="Enojado", width=20, value=1, variable=selected_F, font=("Helvetica", 16))  
    rad21 = Radiobutton(root, text="Feliz", width=20, value=2, variable=selected_F, font=("Helvetica", 16))
    rad31 = Radiobutton(root, text="Neutral", width=20, value=3, variable=selected_F, font=("Helvetica", 16))
    rad41 = Radiobutton(root, text="Triste", width=20, value=4, variable=selected_F, font=("Helvetica", 16))
    rad51 = Radiobutton(root, text="Sorprendido", width=20, value=5, variable=selected_F, font=("Helvetica", 16))
    rad11.grid(column=2, row=1)
    rad21.grid(column=2, row=2)
    rad31.grid(column=2, row=3)
    rad41.grid(column=2, row=4)
    rad51.grid(column=2, row=5)
    Info_11 = Label(root, text= "¿Como te sientes ahora?", bg="green", fg="white", font=("Helvetica", 15))
    Info_11.grid(column=2, row=0, padx=(5, 5), pady=5, sticky="we") 
    btnSiguiente_1 = Button(root, text="Finalizar", width=20, font=("Helvetica", 12), command=finalizar)
    btnSiguiente_1.grid(column=2, row=7, padx=5, pady=5)

def destruir_botones_imagen():
    global rad1, rad2, rad3, rad4, rad5, Info_1, btnSiguiente
    for rad in [rad1, rad2, rad3, rad4, rad5, Info_1, btnSiguiente]:
        rad.destroy()
        
def destruir_botones_musica():
    global rad11, rad21, rad31, rad41, rad51, Info_11, btnSiguiente_1
    for rad in [rad11, rad21, rad31, rad41, rad51, Info_11, btnSiguiente_1]:
        rad.destroy()
          
cap = None
root = Tk()
selected = IntVar()
selected_F = IntVar()
btnIniciar = Button(root, text="Iniciar", width=20, command=iniciar, font=("Helvetica", 12))
btnIniciar.grid(column=0, row=0, padx=5, pady=5, columnspan=2)

lblVideo = Label(root)
lblVideo.grid(column=0, row=1, columnspan=2, rowspan=7)

mpDibujo = mp.solutions.drawing_utils
ConfDibu = mpDibujo.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1)

mpMallaFacial = mp.solutions.face_mesh
MallaFacial = mpMallaFacial.FaceMesh(max_num_faces=3)

root.mainloop()