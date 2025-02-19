import tensorflow as tf
import numpy as np
import cv2
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model

#charger les images depuis un dossier
dataset_dir = "C:/Users/PC ACER/Desktop/Travail/work1/niveau2/archive/train"
output_dir = "C:/Users/PC ACER/Desktop/Travail/work1/niveau2/archive/visageextracted"

haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#Parcourir toutes les images
for class_name in os.listdir(dataset_dir): #parcourt les différents dossiers     
    class_dir = os.path.join(dataset_dir, class_name)
    output_class_dir = os.path.join(output_dir, class_name)
    
    if not os.path.exists(output_class_dir):
        os.makedirs(output_class_dir)
    
    for image_name in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_name)
        image = cv2.imread(image_path)
        
        if image is None:
            continue #passer si l'image ne peut pas être chargée
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(30, 30))
        
        for i, (x, y, w, h) in enumerate(faces):
            face = gray[y:y+h, x:x+w] #extraire le visage
            face_resized = cv2.resize(face, (48, 48)) #Redimensionner à 48*48
            
            output_path = os.path.join(output_class_dir, f"{image_name}_face_{i}.jpg")
            cv2.imwrite(output_path, face_resized) #Sauvegarder le visage extrait

batch_size = 32
img_size = (48, 48)


train_dataset = image_dataset_from_directory(
    output_dir,
    image_size=img_size,
    batch_size=batch_size,
    color_mode="grayscale", #pour l'utilisation de ResNet50
    label_mode="int" 
)

#Convertir le dataset en listes
image_list = []
label_list = []

for images, labels in train_dataset:
    image_list.extend(images.numpy())
    label_list.extend(labels.numpy())
    
#transformer en array numpy
image_list = np.array(image_list)
label_list = np.array(label_list)

#normalisation des pixel (0-255 -> 0-1)
image_list = image_list / 255.0

#Séparation en train (80%) et test (20%)
X_train, X_test, y_train, y_test = train_test_split(image_list, label_list, test_size=0.2, random_state=42)
'''
# Chargeons le modèle pré-entrainé sans la dernière couche
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(48,48,3))

# Gelons les poids du modèle pré-entrainé (optionnel)
base_model.trainable = False

# Ajout des couches 
x = Flatten()(base_model.output)
x = Dense(128, activation="relu")(x)
x = Dense(7, activation="softmax")(x) #car 7 classes de sorties

# Créons le modèle final
model = Model(inputs=base_model.input, outputs=x)

#compilons le modèle
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# entrainons le modèle
model.fit(
    X_train, 
    y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=32
    )

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

#calcule de l'accuracy
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

conf_mat = confusion_matrix(y_test, y_pred_classes)
# Afficher la matrice sous forme de heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=range(len(np.unique(y_test))), yticklabels=range(len(np.unique(y_test))))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Matrice de Confusion")
plt.show()
'''

model = keras.Sequential([
    # Première couche de convolution
    layers.Conv2D(64, (3, 3), activation="relu", input_shape=(48, 48, 1)),
    layers.MaxPooling2D((2, 2)),

    # Deuxième couche de convolution
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Troisième couche de convolution
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Aplatir les données pour les connecter aux couches denses
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.38),  # Évite le surapprentissage
    layers.Dense(7, activation="softmax")  # 7 classes pour les émotions
])

#compilation du modèle
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

#Entrainer le model
model.fit(
    X_train,
    y_train,
    epochs=15,
    validation_data=(X_test, y_test),
    batch_size=32
)
model.save("emotion.h5")
'''
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test accuarcy: {test_accuracy *100:.2f}%")
print(f"Test loss: {test_loss}")

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

conf_matrix = confusion_matrix(y_test, y_pred_classes)

# Afficher la matrice sous forme de heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(len(np.unique(y_test))), yticklabels=range(len(np.unique(y_test))))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Matrice de Confusion")
plt.show()

'''
'''
import matplotlib.pyplot as plt

# Tracer la courbe de l'accuracy et de la loss pendant l'entraînement
plt.figure(figsize=(12, 4))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend(loc='lower right')

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend(loc='upper right')

plt.show()
'''

