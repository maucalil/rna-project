# Bibliotecas
import os
import librosa
import numpy as np
import pandas as pd

# Configurações
audio_dir = "audios"  # Diretório dos arquivos de áudio
output_csv_file = "data.csv"
desired_length = 20  # Duração desejada em segundos
class_mapping = {
    "BA": "Bark",
    "CH": "Chatting",
    "CO": "Construction",
    "FS": "Footsteps",
    "RA": "Rain",
    "TR": "Traffic"
}

audios = os.listdir(audio_dir)
data = []  # Lista para armazenar os dados pré-processados
for audio in audios:
    audio_path = os.path.join(audio_dir, audio)

    # Carrega o áudio
    y, sr = librosa.load(audio_path)

    # Padroniza o comprimento para 20 segundos
    target_length = int(desired_length * sr)
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y))) # completa com 0 se for menor que 20s
    elif len(y) > target_length:
        y = y[:target_length]

    # Extrai as características
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

    # Infere a classe a partir do dicionário de mapeamento
    class_name = class_mapping.get(audio.split("-")[0], "Unknown")

    # Armazena os dados em um dicionário
    features = {
        "audio_name": audio.split(".")[0],
        "class": class_name,
        "spectral_centroid": np.mean(spectral_centroid)
    }

    for i, mfcc in enumerate(mfccs):
        features[f'mfcc_{i + 1}'] = np.mean(mfcc)

    data.append(features)

# Cria um DataFrame do Pandas com os dados
df = pd.DataFrame(data)
df.info()

# Salva o DataFrame em um arquivo CSV
df.to_csv(output_csv_file, index=False)