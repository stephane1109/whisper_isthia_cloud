import streamlit as st
import os
from yt_dlp import YoutubeDL
import whisper
import concurrent.futures
import time


# Fonction pour télécharger l'audio d'une vidéo YouTube
def telecharger_audio_youtube(url, chemin_sortie="Téléchargement"):
    # Télécharge l'audio d'une vidéo YouTube et retourne le chemin du fichier audio téléchargé
    try:
        if not os.path.exists(chemin_sortie):
            os.makedirs(chemin_sortie)
        options_ydl = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(chemin_sortie, '%(title)s.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
        }
        with YoutubeDL(options_ydl) as ydl:
            info = ydl.extract_info(url, download=True)
            chemin_audio = ydl.prepare_filename(info).replace(".webm", ".mp3").replace(".m4a", ".mp3")
            return chemin_audio
    except Exception as e:
        st.error(f"Erreur lors du téléchargement : {e}")
        return None


# Fonction pour transcrire un fichier audio en texte
def transcrire_audio(chemin_audio, taille_modele="base", langue="fr"):
    # Transcrit un fichier audio en texte en utilisant le modèle Whisper
    try:
        modele = whisper.load_model(taille_modele)
        resultat = modele.transcribe(chemin_audio, language=langue)
        return resultat["text"]
    except Exception as e:
        st.error(f"Erreur lors de la transcription : {e}")
        return None


# Interface Streamlit
st.title("Transcription d'audio - Vidéo YouTube ou Fichier MP3")

st.markdown("""
Ce script permet de télécharger l'audio d'une vidéo YouTube ou d'importer un fichier audio MP3, puis de le transcrire en texte.
Veuillez choisir la source de l'audio et saisir les options souhaitées.
""")

# Option de mode débug
debug_mode = st.checkbox("Mode débug", value=False)

# Choix de la source de l'audio
source = st.radio("Choisissez la source de l'audio", options=["URL YouTube", "Fichier audio (mp3)"])

chemin_audio = None

if source == "URL YouTube":
    # Saisie de l'URL de la vidéo YouTube
    url_video = st.text_input("Entrez l'URL de la vidéo YouTube", value="https://www.youtube.com/watch?v=WDQqDOXAUIM")
elif source == "Fichier audio (mp3)":
    # Importer un fichier audio MP3
    fichier_audio = st.file_uploader("Importer un fichier audio MP3", type=["mp3"])

# Sélection de la taille du modèle Whisper
taille_modele = st.selectbox("Choisissez la taille du modèle Whisper",
                             options=["tiny", "base", "small", "medium", "large"], index=1)

# Saisie du code langue pour la transcription (exemple : 'fr' pour français)
langue = st.text_input("Code de la langue pour la transcription", value="fr")

# Bouton pour lancer la transcription
if st.button("Lancer la transcription"):
    if source == "URL YouTube":
        if url_video:
            with st.spinner("Téléchargement de l'audio depuis YouTube..."):
                chemin_audio = telecharger_audio_youtube(url_video)
            if chemin_audio:
                st.success(f"Audio téléchargé : {chemin_audio}")
            else:
                st.error("Le téléchargement de l'audio a échoué.")
        else:
            st.error("Veuillez entrer une URL de vidéo YouTube.")
    elif source == "Fichier audio (mp3)":
        if fichier_audio is not None:
            dossier_sortie = "Téléchargement"
            if not os.path.exists(dossier_sortie):
                os.makedirs(dossier_sortie)
            chemin_audio = os.path.join(dossier_sortie, fichier_audio.name)
            try:
                with open(chemin_audio, "wb") as f:
                    f.write(fichier_audio.getbuffer())
                st.success(f"Fichier audio importé : {chemin_audio}")
            except Exception as e:
                st.error(f"Erreur lors de l'enregistrement du fichier audio : {e}")
                chemin_audio = None
        else:
            st.error("Veuillez importer un fichier audio MP3.")

    # Si le chemin de l'audio est défini, procéder à la transcription avec une barre de progression
    if chemin_audio:
        with st.spinner("Transcription en cours..."):
            # Exécution de la transcription dans un thread séparé pour permettre la mise à jour de la barre de progression
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(transcrire_audio, chemin_audio, taille_modele, langue)
                progress_bar = st.progress(0)
                progress_text = st.empty()  # Zone de texte pour afficher la progression sur une seule ligne
                progress = 0
                # Boucle pour simuler la progression
                while not future.done():
                    time.sleep(0.1)  # Pause de 0.1 seconde
                    progress = min(99, progress + 1)
                    progress_bar.progress(progress)
                    if debug_mode:
                        progress_text.text(f"Progression : {progress}%")
                texte_transcription = future.result()
                progress_bar.progress(100)
                progress_text.text("Progression : 100%")
        if texte_transcription:
        # Affichage de la transcription dans l'interface
            st.subheader("Transcription")
            st.text_area("Texte de la transcription", texte_transcription, height=300)
            # Bouton de téléchargement de la transcription
            try:
                with open(chemin_transcription, "r", encoding="utf-8") as fichier:
                    contenu_transcription = fichier.read()
                st.download_button("Télécharger la transcription",
                                   data=contenu_transcription,
                                   file_name=os.path.basename(chemin_transcription),
                                   mime="text/plain")
            except Exception as e:
                st.error(f"Erreur lors de la préparation du téléchargement : {e}")
        else:
            st.error("La transcription a échoué.")
