# utils.py

import os
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import random
from tqdm.notebook import tqdm  # barre de progression
import pickle
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

top_k = 5000 # Chosir les 5000 mots les plus frequents du vocabulaire
embedding_dim = 256 # Dimension de l'espace d'embedding
units = 512 # Taille de la couche caché dans le RNN
vocab_size = top_k + 1 # Taille du vocabulaire (5000 mots + 1 pour le token <pad>)

class CNN_Encoder(tf.keras.Model):
    # Comme les images sont déjà prétraités par InceptionV3 est représenté sous forme compacte
    # L'encodeur CNN ne fera que transmettre ces caractéristiques à une couche dense
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        self.fc = tf.keras.layers.Dense(embedding_dim)
        self.embedding_dim = embedding_dim

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embedding_dim": self.embedding_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.units = units
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = self.V(tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = tf.reduce_sum(attention_weights * features, axis=1)
        return context_vector, attention_weights
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim  # ✅ Fix ajouté

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.units)


    def call(self, x, features, hidden):
        # (batch_size, 256), (batch_size, 64, 256)
        context_vector, attention_weights = self.attention(features, hidden)

        # (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # Concaténer context_vector à chaque embedding (batch_size, 1, embedding_dim + context_dim)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # Passer dans la GRU
        output, state = self.gru(x)

        x = self.fc1(output)  # (batch_size, 1, units)
        x = tf.reshape(x, (-1, x.shape[2]))  # (batch_size, units)
        x = self.fc2(x)  # (batch_size, vocab_size)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embedding_dim": self.embedding_dim,
            "units": self.units,
            "vocab_size": self.vocab_size
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)   

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
def load_autoencoder(model_path, learning_rate=0.001):
    """
    Fonction pour charger l'autoencodeur depuis un fichier HDF5.

    :param weight_path: Chemin vers la sauvegarde du modèle
    :param learning_rate: Taux d'apprentissage pour l'optimiseur Adam (par défaut 0.001)

    :return: Modèle de l'autoencodeur

    """
    # # Chargement des poids
    if not os.path.exists(model_path):
        print("⚠️ Le fichier de poids n'existe pas.")

    # autoencoder.load_weights(weight_path)
    autoencoder = keras.models.load_model(model_path, custom_objects={'mse': keras.losses.MeanSquaredError()})
    print(f"Autoencodeur chargé depuis {model_path}")
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
    try :
        denoised_images = autoencoder.predict(images, verbose=0)
        print("Prédiction réussie.")
    except Exception as e:
        print(f"Erreur lors de la conversion des images : {e}")
        print("Prédiction échouée.")
        return None
    
    return denoised_images

#---------------------------------------------------------------
def plot_denoised_examples(noisy_images, denoised_images, n=5):
    """
    Affiche des exemples d'images bruitées et débruitées côte à côte.
    """
    plt.figure(figsize=(12, 6), dpi=100)
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
        denoised_img = (denoised_img - denoised_img.min()) / (denoised_img.max() - denoised_img.min() + 1e-8)
        plt.imshow(denoised_img, cmap=cmap_mode)
        plt.title("Débruitée")
        plt.axis('off')

    plt.tight_layout()
    plt.show()
#---------------------------------------------------------------
def load_model_caption(token_dir, save_dir):
    """
    Fonction pour charger le modèle de légende depuis un fichier HDF5.

    :param model_path: Chemin vers la sauvegarde du modèle
    """
    with open(os.path.join(token_dir, 'tokenizer.pickle'), 'rb') as handle:
        tokenizer = pickle.load(handle)

    # === Initialisation du modèle et chargement des poids ===
    Load_encoder = CNN_Encoder(embedding_dim)
    Load_decoder = RNN_Decoder(embedding_dim, units, vocab_size)
    Load_optimizer = tf.keras.optimizers.Adam()

    ckpt = tf.train.Checkpoint(encoder=Load_encoder, decoder=Load_decoder)
    ckpt_manager = tf.train.CheckpointManager(ckpt, save_dir, max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
        status = ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print(f"Checkpoint restauré depuis {ckpt_manager.latest_checkpoint}")
    else:
        print("Aucun checkpoint trouvé.")

    image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output
    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    return Load_encoder, Load_decoder, image_features_extract_model, tokenizer

def generate_caption_from_array(image_array, Load_encoder, Load_decoder, image_features_extract_model, tokenizer, max_length):
    hidden = Load_decoder.reset_state(batch_size=1)
    
    img = tf.image.resize(image_array, (299, 299))  # Requis pour InceptionV3
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    temp_input = tf.expand_dims(img, 0)

    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))
    features = Load_encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for _ in range(max_length):
        predictions, hidden, _ = Load_decoder(dec_input, features, hidden)
        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        word = tokenizer.index_word.get(predicted_id, '')
        if word == '<end>':
            break
        result.append(word)
        dec_input = tf.expand_dims([predicted_id], 0)

    return ' '.join(result)
    
#---------------------------------------------------------------
def plot_captionned_images(images_denoised, captions, n=5):
    """
    Affiche un nombre limité d'images avec leurs légendes.

    Args:
        image_paths (list): Liste des chemins d'images.
        captions (list): Liste des légendes correspondantes.
        n (int): Nombre d'images à afficher.
    """
    # Affichage des images et de leur descriptions
    plt.figure(figsize=(20, 20))
    for idx, image_array  in enumerate(images_denoised):
    
        plt.subplot(5, 2, idx + 1)
        plt.imshow(image_array)
        plt.axis('off')
        plt.title(captions[idx], fontsize=10)

    plt.tight_layout()
    plt.show()