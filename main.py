# pages/analyse_mouvement.py
# Analyse des mouvements par flux optique (OpenCV) avec extraction préalable des frames via FFmpeg.
# Corrige le "noir" dû à VideoCapture en environnement headless.
# Hypothèse : main.py a préparé la vidéo et son chemin est dans st.session_state["video_base"].

import math
import shutil
from pathlib import Path

import numpy as np
import streamlit as st

from core_media import initialiser_repertoires, info_ffmpeg

# =========================
# Utilitaires généraux
# =========================

def _ffmpeg_path() -> str | None:
    """Retourne le chemin de ffmpeg ou None."""
    chemin, _ = info_ffmpeg()
    return chemin

def _run(cmd: list[str]) -> tuple[bool, str]:
    """Exécute une commande système, renvoie (ok, log)."""
    import subprocess
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        out = (res.stdout or "").strip()
        err = (res.stderr or "").strip()
        log = "\n".join([s for s in (out, err) if s]).strip()
        return True, log
    except subprocess.CalledProcessError as e:
        out = (e.stdout or "").strip()
        err = (e.stderr or "").strip()
        log = "\n".join([s for s in (out, err) if s]).strip() or str(e)
        return False, log
    except Exception as e:
        return False, f"Erreur d'exécution : {e}"

def _charger_cv2():
    """Import différé d'OpenCV (opencv-python-headless recommandé)."""
    try:
        import cv2  # type: ignore
        return cv2, None
    except Exception as e:
        return None, f"OpenCV introuvable : {e}. Ajoute 'opencv-python-headless' à requirements.txt."

def _extraire_frames_ffmpeg(ff: str, video: Path, dossier: Path, fps_ech: float, largeur: int) -> tuple[bool, str]:
    """
    Extrait des frames JPEG avec FFmpeg à cadence régulière.
    - fps_ech : cadence d’échantillonnage (ex. 5 i/s)
    - largeur : redimensionnement (ex. 640) pour accélérer le flux optique
    """
    if dossier.exists():
        try:
            shutil.rmtree(dossier)
        except Exception:
            pass
    dossier.mkdir(parents=True, exist_ok=True)
    motif = str(dossier / "frame_%06d.jpg")

    # On force un chemin simple, échantillonnage fps, resize et qualité correcte.
    filtre = f"fps={fps_ech},scale={largeur}:-2"
    cmd = [
        ff, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video),
        "-vf", filtre,
        "-q:v", "2",
        motif
    ]
    return _run(cmd)

def _charger_images_en_gris(cv2, dossier: Path) -> list[np.ndarray]:
    """Charge toutes les frames JPG du dossier en niveaux de gris (retourne une liste d'images)."""
    images = []
    files = sorted(dossier.glob("frame_*.jpg"))
    for f in files:
        arr = cv2.imdecode(np.fromfile(str(f), dtype=np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            continue
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        images.append(gray)
    return images

def _charger_images_rgb_pour_vignettes(cv2, dossier: Path) -> list[np.ndarray]:
    """Charge les mêmes images en RGB pour affichage des vignettes."""
    images = []
    files = sorted(dossier.glob("frame_*.jpg"))
    for f in files:
        arr = cv2.imdecode(np.fromfile(str(f), dtype=np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            continue
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        images.append(rgb)
    return images

def _flux_farneback(cv2, prev_gray: np.ndarray, gray: np.ndarray):
    """Flux optique dense (Farneback)."""
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray,
        None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2,
        flags=0
    )
    return flow

def _metriques_sur_mag(mag: np.ndarray, seuil_pix: float) -> dict[str, float]:
    """Calcule les métriques de variation à partir de la magnitude du flux."""
    m = float(np.mean(mag))
    s = float(np.std(mag))
    p95 = float(np.percentile(mag, 95))
    ratio_mobile = float(np.mean(mag > seuil_pix))
    energie = float(np.sum(mag))
    return {
        "magnitude_moyenne": m,
        "magnitude_ecart_type": s,
        "magnitude_p95": p95,
        "ratio_pixels_mobiles": ratio_mobile,
        "energie_mouvement": energie,
    }

# =========================
# Page Streamlit
# =========================

BASE_DIR, REP_SORTIE, REP_TMP = initialiser_repertoires()

st.set_page_config(page_title="Analyse des mouvements (flux optique)", layout="wide")
st.title("Analyse des mouvements (flux optique)")
st.markdown("**www.codeandcortex.fr**")

# Contrôles d'entrée
if not st.session_state.get("video_base"):
    st.warning("Aucune vidéo préparée. Va d’abord sur la page d’accueil pour préparer la source.")
    st.stop()

video_path = Path(st.session_state["video_base"])
if not video_path.exists():
    st.error("La vidéo préparée est introuvable sur le disque.")
    st.stop()

ff = _ffmpeg_path()
if not ff:
    st.error("FFmpeg introuvable. Fournis un binaire ./bin/ffmpeg ou vérifie /usr/bin/ffmpeg.")
    st.stop()

cv2, err = _charger_cv2()
if cv2 is None:
    st.error(err)
    st.stop()

st.subheader("Paramètres")
c1, c2, c3 = st.columns(3)
with c1:
    fps_ech = st.number_input("Cadence d’échantillonnage (i/s)", min_value=1, max_value=30, value=5, step=1)
with c2:
    largeur_det = st.selectbox("Largeur des frames d’analyse", [480, 640, 960], index=1)
with c3:
    seuil_pix = st.number_input("Seuil pixel 'mobile' (px/frame)", min_value=0.1, max_value=10.0, value=1.5, step=0.1)

c4, c5 = st.columns(2)
with c4:
    nb_vignettes = st.number_input("Nombre de vignettes à afficher", min_value=8, max_value=200, value=40, step=4)
with c5:
    afficher_log = st.checkbox("Afficher le journal FFmpeg", value=False)

# Extraction des frames via FFmpeg
frames_dir = (BASE_DIR / "frames_analysis" / video_path.stem).resolve()

ok_ext, log_ext = _extraire_frames_ffmpeg(ff, video_path, frames_dir, float(fps_ech), int(largeur_det))
if not ok_ext:
    st.error("Échec extraction des frames avec FFmpeg.")
    if afficher_log:
        st.code(log_ext or "(log vide)", language="bash")
    st.stop()

# Chargement des images pour analyse et vignettes
imgs_gray = _charger_images_en_gris(cv2, frames_dir)
imgs_rgb = _charger_images_rgb_pour_vignettes(cv2, frames_dir)

if len(imgs_gray) < 2:
    st.error("Trop peu de frames extraites pour analyser le mouvement.")
    if afficher_log:
        st.code(log_ext or "(log vide)", language="bash")
    st.stop()

# Boucle de flux optique + métriques
metriques = []
energies = []
for i in range(1, len(imgs_gray)):
    flow = _flux_farneback(cv2, imgs_gray[i-1], imgs_gray[i])
    mag = np.linalg.norm(flow, axis=2)
    m = _metriques_sur_mag(mag, float(seuil_pix))
    metriques.append(m)
    energies.append(m["energie_mouvement"])

# Détection de pics simples (moyenne + 2σ)
energies = np.array(energies, dtype=float)
moy = float(np.mean(energies))
std = float(np.std(energies))
seuil_pic = moy + 2.0 * std
pics_idx = [i+1 for i, e in enumerate(energies) if e >= seuil_pic]  # +1 car energies commence à la 2e frame

st.subheader("Métriques globales")
st.write(f"Énergie moyenne du mouvement : {moy:.2f}")
st.write(f"Écart-type de l’énergie : {std:.2f}")
st.write(f"Seuil de détection de pics (moy + 2σ) : {seuil_pic:.2f}")
if pics_idx:
    # Convertir en positions dans la séquence extraite (i/fps_ech)
    temps_pics = [idx / float(fps_ech) for idx in pics_idx]
    st.write("Pics détectés (approx.) : " + ", ".join([f"t≈{t:.1f}s (#{idx})" for t, idx in zip(temps_pics, pics_idx)]))
else:
    st.write("Aucun pic détecté au seuil courant.")

# Export CSV
import pandas as pd
lignes = []
for i, m in enumerate(metriques, start=1):
    lignes.append({
        "index_sequence": i,  # index dans la séquence de frames extraites
        "temps_s_approx": i / float(fps_ech),
        **m
    })
df = pd.DataFrame(lignes)
csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button("Télécharger les métriques (CSV)", data=csv_bytes, file_name="metriques_flux_optique.csv", mime="text/csv")

# Affichage de vignettes uniformément réparties
st.subheader("Vignettes réparties sur la vidéo")
N = len(imgs_rgb)
if N == 0:
    st.info("Aucune vignette disponible.")
else:
    # Indices à afficher (uniformément répartis)
    nb = int(nb_vignettes)
    idxs = np.linspace(0, N - 1, num=nb, dtype=int)
    cols_par_ligne = 8
    lignes = math.ceil(len(idxs) / cols_par_ligne)
    k = 0
    for _ in range(lignes):
        cols = st.columns(cols_par_ligne)
        for c in cols:
            if k >= len(idxs):
                break
            i = int(idxs[k])
            c.image(imgs_rgb[i], caption=f"#{i} • t≈{i/float(fps_ech):.1f}s", use_container_width=False)
            k += 1

# Journal FFmpeg optionnel
if afficher_log:
    with st.expander("Journal FFmpeg"):
        st.code(log_ext or "(log vide)", language="bash")
