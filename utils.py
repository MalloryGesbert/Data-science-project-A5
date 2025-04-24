# utils.py
import os

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import shutil
import random
import kaggle
import requests
from zipfile import ZipFile
from tqdm.notebook import tqdm  # barre de progression

# from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.utils import img_to_array, load_img
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from PIL import Image

#---------------------------------------------------------------
def get_infos_datas(dataset_path):
    """
    Fonction pour obtenir les informations sur le dataset.
    """
    num_images = sum([len(files) for _, _, files in os.walk(dataset_path)])
    print(f"Nombre total d'images : {num_images}")

    return num_images

 #---------------------------------------------------------------   
def display_random_images(dataset_path, num_images=5):
    """
    Affiche un nombre aléatoire d'images d'un dossier.
    
    :param dataset_path: Chemin vers le dataset
    :param num_images: Nombre d'images à afficher (par défaut 5)
    """
    # Collecte de toutes les images dans les sous-dossiers
    image_paths = []
    for img_name in os.listdir(dataset_path):
        if img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff")):
            image_paths.append(os.path.join(dataset_path,img_name))
    
    # Sélection aléatoire sans doublons
    if num_images > len(image_paths):
        raise ValueError(f"Seulement {len(image_paths)} images disponibles, impossible d’en afficher {num_images}.")
    
    selected_paths = random.sample(image_paths, num_images)

    # Affichage
    fig, axes = plt.subplots(1, num_images, figsize=(4 * num_images, 4))
    if num_images == 1:
        axes = [axes]
    for ax, img_path in zip(axes, selected_paths):
        img = PIL.Image.open(img_path)
        ax.imshow(img)
        width, height = img.size
        ax.set_title(f"{os.path.basename(img_path)}\n{height}x{width}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()

#---------------------------------------------------------------
def clean_images_dataset(dataset_path_arg):
    """
    Fonction pour nettoyer le dataset en supprimant les fichiers corrompus ou non images.

    :param dataset_path_arg: Chemin vers le dataset
    """
    # Dictionnaire pour stocker le nombre d'images corrompues par classe
    corrupted_count = 0
    dataset_path = dataset_path_arg
    print("Début de la vérification des images ...")

    # Récupération de toutes les images pour calculer la progression
    all_files = []
    for file_name in os.listdir(dataset_path): 
        all_files.append((dataset_path, file_name))

    total_files = len(all_files)
    checked_files = 0  # Pour la progression

    # Parcours des images avec affichage de la progression
    for dataset_path, file_name in all_files:
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            try:
                with open(os.path.join(dataset_path, file_name), 'rb') as file:
                    img_bytes = file.read()  # Lire les bytes de l'image
                    img = tf.image.decode_image(img_bytes)  # Essayer de décoder l'image
            except Exception as e:
                corrupted_count += 1
                print(f"\nImage corrompue : {file_name}. Exception: {e}")
                os.remove(os.path.join(dataset_path, file_name))
                print(f"Image {file_name} supprimée.")
        else:
            corrupted_count += 1
            print(f"\nLe fichier {file_name} n'est pas une image.")
            os.remove(os.path.join(dataset_path, file_name))
            print(f"Fichier {file_name} supprimé.")

        # Mise à jour de la progression
        checked_files += 1
        progress = (checked_files / total_files) * 100
        print(f"\rProgression : [{int(progress)}%] {checked_files}/{total_files} images vérifiées", end="")

    print("\nVérification des fichiers terminée.")

    # Nombre total d'images corrompues
    print(f"Nombre total d'images corrompues ou non image : {corrupted_count}")

#---------------------------------------------------------------
def resize_images_dataset(dataset_path_arg, target_size=(256, 256)):
    """
    Fonction pour redimensionner toutes les images d'un dataset à une taille cible.
    
    :param dataset_path_arg: Chemin vers le dataset
    :param target_size: Taille cible (largeur, hauteur) par défaut (128, 128)
    """
    dataset_path = dataset_path_arg
    print("Début du redimensionnement des images ...")

    # Récupération de toutes les images pour calculer la progression
    all_files = []
    for file_name in os.listdir(dataset_path): 
        all_files.append((dataset_path, file_name))

    total_files = len(all_files)
    resized_count = 0  # Pour la progression

    # Parcours des images avec affichage de la progression
    for dataset_path, file_name in all_files:
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            try:
                img = load_img(os.path.join(dataset_path, file_name), target_size=target_size)  # Charger et redimensionner l'image
                img.save(os.path.join(dataset_path, file_name))  # Sauvegarder l'image redimensionnée
                resized_count += 1
            except Exception as e:
                print(f"\nErreur lors du redimensionnement de {file_name}. Exception: {e}")

        # Mise à jour de la progression
        progress = (resized_count / total_files) * 100
        print(f"\rProgression : [{int(progress)}%] {resized_count}/{total_files} images redimensionnées", end="")

    print("\nRedimensionnement des fichiers terminé.")

#---------------------------------------------------------------
def load_model(model_path):
    """
    Fonction pour charger notre modèle depuis un fichier HDF5.

    :param model_path: Chemin vers la sauvegarde du modèle
    """
    model = keras.models.load_model(model_path)
    print(f"Modèle chargé depuis {model_path}")
    return model

#---------------------------------------------------------------
def get_predicted_photos(model, dataset_path):
    """
    Parcourt toutes les images du dossier, prédit leur catégorie et retourne
    une liste des chemins des images classées comme 'Photo'.

    Args:
        model (keras.Model): Le modèle de prédiction.
        dataset_path (str): Le chemin vers le dossier contenant les images.

    Returns:
        list: Liste des chemins d'images prédites comme étant de type 'Photo'.
    """
    class_names = ['Painting', 'Photo', 'Schematics', 'Sketch', 'Text']
    photo_paths = []

    # Liste de toutes les images valides dans le dossier
    image_files = [f for f in os.listdir(dataset_path)
                   if os.path.isfile(os.path.join(dataset_path, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    print(f"Traitement de {len(image_files)} images...")

    for image_name in tqdm(image_files, desc="Prédiction en cours", unit="image"):
        img_path = os.path.join(dataset_path, image_name)
        img = load_img(img_path, target_size=(180, 180))  # ou l'image_size utilisée lors de l'entraînement
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array, verbose=0)
        scores = tf.nn.softmax(prediction[0])
        predicted_label = class_names[np.argmax(scores)]

        if predicted_label == "Photo":
            photo_paths.append(img_path)

    print(f"\n✅ {len(photo_paths)} images classées comme 'Photo'.")
    return photo_paths

#---------------------------------------------------------------
def display_images_with_predictions(model, dataset_path, num_images=5):
    """
    Affiche un nombre limité d'images avec la prédiction du modèle.

    Args:
        model (keras.Model): Le modèle de prédiction.
        dataset_path (str): Le chemin vers le dossier contenant les images.
        num_images (int): Le nombre d'images à afficher.
    """
    class_names = ['Painting', 'Photo', 'Schematics', 'Sketch', 'Text']
    
    # Liste de toutes les images valides dans le dossier
    image_files = [f for f in os.listdir(dataset_path)
                   if os.path.isfile(os.path.join(dataset_path, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    if num_images > len(image_files):
        raise ValueError(f"Seulement {len(image_files)} images disponibles, impossible d’en afficher {num_images}.")

    # Sélection aléatoire sans doublons
    selected_images = random.sample(image_files, num_images)

    # Affichage
    ncols = num_images
    nrows = 1
    plt.figure(figsize=(ncols * 4, 4))

    for i, image_name in enumerate(selected_images):
        img_path = os.path.join(dataset_path, image_name)
        img = load_img(img_path, target_size=(180, 180))  # Adapter si besoin
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Ajouter batch dimension

        prediction = model.predict(img_array, verbose=0)
        scores = tf.nn.softmax(prediction[0])
        predicted_label = class_names[np.argmax(scores)]
        confidence = 100 * np.max(scores)
        
        # Affichage
        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(img)
        plt.title(f"{predicted_label}\n{confidence:.2f}%")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

#---------------------------------------------------------------
def load_autoencoder(weight_path, learning_rate=0.001):
    """
    Fonction pour charger l'autoencodeur depuis un fichier HDF5.

    :param weight_path: Chemin vers la sauvegarde du modèle
    :param learning_rate: Taux d'apprentissage pour l'optimiseur Adam (par défaut 0.001)

    :return: Modèle de l'autoencodeur

    """
    # Encoder
        # Entrée de l'image
    input_img = Input(shape=(256, 256, 3))  # à adapter selon ta taille d’image
        # Bloc 1
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
        # Bloc 2
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    # Decoder
        # Bloc 1
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
        # Bloc 2
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
        # Dernière couche (sortie)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    # Création du modèle générique
    autoencoder = Model(input_img, decoded)

    # Compilation du modèle générique
    autoencoder.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['accuracy'])

    # Chargement des poids
    autoencoder.load_weights(weight_path)

    print(f"Autoencodeur chargé depuis {weight_path}")
    return autoencoder

#---------------------------------------------------------------
def load_images_from_paths(image_paths, target_size=(256, 256)):
    """
    Charge et convertit en array une liste d'images depuis leurs chemins.

    Args:
        image_paths (list of str): Chemins vers les fichiers image.
        target_size (tuple): Taille de redimensionnement des images.

    Returns:
        list of np.ndarray: Images chargées sous forme de tableaux numpy.
    """
    images = []
    for path in image_paths:
        img = Image.open(path)
        # Converti en RGB si necessaire pour assurer la coherence
        if img.mode != 'RGB':
            img = img.convert('RGB')
        images.append(np.array(img))
    return images

def resize_images(images, target_size=(256,256)):
    """Redimensionne toutes les images et les convertit en RGB"""
    resized_images = []
    for img in images:
        img_pil = Image.fromarray(img).convert('RGB')  # Forcer en RGB
        img_resized = img_pil.resize(target_size)
        resized_images.append(np.array(img_resized))
    return np.array(resized_images)

def preprocess_data(images):
    """Normalisation des images"""
    images = np.array(images)
    images = images.astype('float32') / 255.  # Normalisation
    return images
#---------------------------------------------------------------
def denoise_images(autoencoder, images):
    """
    Débruite une liste d'images à l'aide d'un autoencodeur.

    Args:
        autoencoder (keras.Model): Le modèle d'autoencodeur entraîné pour débruiter les images.
        images (list or np.ndarray): Liste ou tableau de numpy arrays représentant les images (non normalisées).

    Returns:
        np.ndarray: Tableau des images débruitées, de même forme que les images d'entrée.
    """
    denoised_images = autoencoder.predict(images, verbose=0)
    print("Prédiction réussie.")
    return denoised_images


#---------------------------------------------------------------
def plot_denoised_examples(noisy_images, denoised_images, n=5):
    """
    Affiche des exemples d'images bruitées et débruitées côte à côte.
    """
    plt.figure(figsize=(15, 4))
    for i in range(n):
        cmap_mode = 'gray' if noisy_images[i].shape[-1] == 1 else None

        # Image bruitée
        plt.subplot(2, n, i + 1)
        noisy_img = noisy_images[i].squeeze()
        plt.imshow(noisy_img, cmap=cmap_mode)
        plt.title("Bruitée")
        plt.axis('off')

        # Image débruitée
        plt.subplot(2, n, i + 1 + n)
        denoised_img = denoised_images[i].squeeze()
        # denoised_img = (denoised_img - denoised_img.min()) / (denoised_img.max() - denoised_img.min() + 1e-8)
        plt.imshow(denoised_img, cmap=cmap_mode)
        plt.title("Débruitée")
        plt.axis('off')

    plt.tight_layout()
    plt.show()