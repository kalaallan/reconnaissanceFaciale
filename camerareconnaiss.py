import tensorflow as tf
import cv2
import numpy as np
from collections import deque

#Buffer pour stocker les dernières prédictions (taille ajustable)
buffer_size = 10
prediction_buffer = deque(maxlen=buffer_size)

def get_stable_prediction(new_prediction):
    """Ajoute la nouvelle prédiction au buffer et retourne la plus fréquente."""
    prediction_buffer.append(new_prediction)
    return max(set(prediction_buffer), key=prediction_buffer.count)

#charger le modèle entrainé
model = tf.keras.models.load_model("emotion.h5")

#definition des labels émotions
emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

#charger le détecteur de visages haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#ouvrir la webcam
cam = cv2.VideoCapture(0) #pour la webcam intégrée

while True:
    #lire une image depuis la webcam
    ret, frame = cam.read()
    if not ret:
        break # si la capture échoue on sort de la boucle
    
    # convertissons en niveaux de gris 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #detection des visages
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3, minSize=(48, 48))
    
    for (x, y, w, h) in faces:
        #extraire le visage
        face = gray[y:y+h, x:x+w]
        
        #redimentionner au format attendu par le modèle
        face = cv2.resize(face, (48, 48))
        face = np.expand_dims(face, axis=0) #ajouter une dimension batch
        face = np.expand_dims(face, axis=-1) #ajouter le canal 
        face = face / 255.0
        
        #prediction de l'émotion
        prediction = model.predict(face)
        emotion_index = np.argmax(prediction) #avoir l'index de la prédiction
        emotion = emotion_labels[emotion_index]
        
        #dessinons un rectangle autour du visage et afficher l'émotion
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        stable_prediction = get_stable_prediction(emotion)
        cv2.putText(frame, stable_prediction, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
    # Affichons le flux vidéo avec détection
    cv2.imshow("Detection des emotions", frame)
    
    # quittons avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# Fermons la webcam et les fenêtres OpenCV
cam.release()
cv2.destroyAllWindows()
    