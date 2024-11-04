import os
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader

from model import SoundNet
import torch
import librosa
import tqdm as tqdm
import numpy as np
import random 

def _extract_all_files(folder_path: Path, format: str):
    files = [] 
    for dirpath, _, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.endswith(format): 
                full_path = os.path.join(dirpath, filename) 
                files.append(full_path)
    return files

def _extract_df_attributes(folder_path: Path):
    csv_files = [
        os.path.join(dirpath, filename) 
        for dirpath, _, filenames in os.walk(folder_path) 
        for filename in filenames 
        if filename.endswith('.csv')
    ]
    if not csv_files:
        raise ValueError("No labels provided in .csv")
    
    df_attributes = pd.read_csv(csv_files[0]) 
    
    return df_attributes

def _normalize(data):
    data = np.array(data, dtype=np.float32)  # Convert to float32 to ensure numeric operations
    norm_data = data / np.max(np.abs(data)) * 256
    return norm_data

def _extract_song_audio_list(audio_files, song_files, df):
    songs = ['Potter', 'StarWars', 'Panther', 'Rain', 'Hakuna', 'Mamma','Showman','Frozen']
    song_name = [df.loc[df['Public filename'] == file.split('/')[-1], 'Song'].values 
                  for file in audio_files]
    song_indices = [songs.index(name[0]) for name in song_name]
    organised_song_audio = [song_files[i] for i in song_indices]
    return organised_song_audio
    
def _create_dataframe(audios_path: Path, song_path: Path, folds: int):
    song_files = sorted(_extract_all_files(song_path, ".mp3"))
    audio_files = _extract_all_files(audios_path, ".wav")
    
    df = _extract_df_attributes(audios_path)
    organised_songs = _extract_song_audio_list(audio_files, song_files, df)
    
    data = pd.DataFrame(columns=["audio", "song"], data=zip(audio_files, organised_songs))
    
    data["fold"] = range(0, len(data))  # Number from 1 to len(data) -1
    data.fold = (data.fold % folds) + 1  # Asigning fold to each row

    return data
    
class TripletDataset:
    def __init__(
        self,
        audios,
        songs,
        fs,
        model_path,
        preprocess,
        ) -> None:
        
        self.audios = audios
        self.songs = songs
        self.fs = fs
        self.model_path = model_path
        self.preprocess = preprocess
    
    def __getitem__(self, i):
        positive_audio = self.audios[i]
        anchor_song = self.songs[i]

        negative_index = random.choice(
            [j for j, song in enumerate(self.songs) if song != anchor_song]
        )
        
        positive_audio_data, _ = librosa.load(positive_audio, sr=self.fs)
        negative_audio_data, _ = librosa.load(self.audios[negative_index], sr=self.fs)
        
        anchor_song_data, _ = librosa.load(anchor_song, sr=self.fs)
        if self.preprocess:
            positive_audio_data = _normalize(positive_audio_data)
            negative_audio_data = _normalize(negative_audio_data)
            anchor_song_data = _normalize(anchor_song_data)
        
        anchor_song_data = torch.tensor(anchor_song_data).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        print(anchor_song_data.shape)
        encoder = SoundNet()
        if self.model_path:
            encoder.load_state_dict(torch.load(self.model_path))
        with torch.no_grad():
            song_embed, _, _ = encoder(anchor_song_data)

        return positive_audio_data, negative_audio_data, song_embed

    def __len__(self):
        return len(self.audios)
        
def get_dataloaders(config):
    fs = config.get("dataset").get("fs", None)
    folds = config.get("folds")
    
    data = _create_dataframe(audios_path=config.get("dataset").get("hw_files_path"), 
                            song_path=config.get("dataset").get("songs_path"),
                            folds=folds)
    
    valid_fold = 1 # ! Change here
    train_subset = data.loc[data.fold != valid_fold]
    valid_subset = data.loc[data.fold == valid_fold]
    
    train_dataset = TripletDataset(
        train_subset.audio.to_list(),
        train_subset.song.to_list(),
        fs=fs,
        model_path=config.get("model").get("path", None),
        preprocess=config.get("dataset").get("preprocess", None),
    )
    
    valid_dataset = TripletDataset(
        valid_subset.audio.to_list(),
        valid_subset.song.to_list(),
        fs=fs,
        model_path=config.get("model").get("path", None),
        preprocess=config.get("dataset").get("preprocess", None),
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.get("batch_size"), shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.get("batch_size"), shuffle=False)
    
    return train_loader, valid_loader