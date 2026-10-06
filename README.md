# Predicting Hit Songs (CS229 paper re-implementation, own dataset)

## Setup
    pip install -r requirements.txt

## Dataset
Download **Spotify Hit Predictor Dataset** from Kaggle (file `dataset-of-10s.csv`,
6,398 songs, balanced 3,199 hit / 3,199 non-hit) and place it in `data/`.
Columns: track, artist, uri, danceability, energy, key, loudness, mode, speechiness,
acousticness, instrumentalness, liveness, valence, tempo, duration_ms,
time_signature, chorus_hit, sections, target.

## Run
    python src/train_models.py --data data/dataset-of-10s.csv   # trains 8 models, saves best
    python src/predict.py --input data/some_songs.csv           # 1 = hit, 0 = non-hit

## Optional: raw-audio pipeline (paper's approach)
    python src/extract_features.py --audio_dir audio/ --labels labels.csv
Produces the 518 chorus features (11 librosa features x 7 stats). You can feed this
CSV to train_models.py too (drop `filename`/`chorus_found` first or add them to ID_COLS).

## Method
75/25 stratified split, StandardScaler, 5-fold CV GridSearch (scoring = F1),
best model chosen by CV F1; reports accuracy, precision, recall, F1.
