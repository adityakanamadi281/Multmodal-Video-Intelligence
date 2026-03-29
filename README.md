# Multimodal Video Intelligence 


## 📌 Project Overview
An end-to-end machine learning pipeline designed to process unstructured video clips (Reels/Shorts) and extract meaningful, structured signals. By leveraging efficient, physically interpretable heuristics alongside unsupervised clustering, this system analyzes raw video without requiring predefined labels. 

The pipeline is modeled to extract metrics evaluating structural edits, visual density, audio characteristics, and commercial intent, serving as a highly scalable solution for content discovery, moderation, and recommendation systems.

## 🚀 Key Features

### 1. Visual Signals Extraction
- **Editing Intensity**: Calculates scene cuts per second to differentiate heavily edited promotional content from simpler vlogs.
- **Visual Quality**: Estimates quality using a brightness-contrast proxy layer (differentiating amateur vs. studio-quality videos).
- **Face Detection**: Tracks the presence of people (`haarcascade_frontalface_default`), aiding in classifying vlogs vs. product B-roll.
- **Text Overlay Proxy**: Measures high-frequency edge density in lower-thirds to detect heavy text/logo usage.

### 2. Audio Signals Extraction
- **Audio Presence**: Detects whether an active audio track exists.
- **Audio Energy**: Measures structural volume/loudness using mean RMS amplitude.
- **Speech Proxy (ZCR)**: Uses Zero-Crossing Rate to identify conversational patterns and voiceovers vs. flat background music.

### 3. Synthesized Archetypes & Commercial Intent
Calculates a heuristic score (`commercial_intent_score`) blending high text density, aggressive cuts, and loud audio volumes to intelligently classify videos into archetypes:
- `High-Energy Ad/Promo`
- `Vlog/Monologue`
- `Silent B-Roll`
- `Mixed Context`

### 4. Unsupervised Pattern Clustering
Utilizes **K-Means Clustering** across all normalized visual and audio features to automatically group similar unlabelled videos into organic behavioral clusters.

## 🧰 Tech Stack
- **Language:** Python 3
- **Computer Vision:** OpenCV (`cv2`)
- **Audio Processing:** Librosa
- **Video Manipulation:** MoviePy
- **Machine Learning:** Scikit-Learn (`KMeans`, `StandardScaler`)
- **Data Science:** Pandas, NumPy, tqdm

## 🏗️ Scalability & Production Architecture
The project inherently proposes a scalable design for massive video ingestion (1M+ videos):
1. **Serverless Infrastructure:** Highly parallelized ingestion where videos uploaded to Object Stores (e.g., S3, Cloud Storage) trigger independent pipeline executions.
2. **Tiered Inference Cascade:** This pipeline acts as a lightweight gating mechanism—only routing videos with high `commercial_intent_score` to expensive, GPU-backed deep-learning inferences (e.g., Whisper, CLIP).
3. **Zero-Shot Generalization:** Built completely upon foundational geometric and acoustic properties, providing absolute out-of-the-box cross-lingual and cross-demographic compatibility without retraining.

## 🛠️ Setup & Usage

Since this solution was developed for a Kaggle environment (`sparkonomy-ml-hackathon`), it requires basic dataset structural alignment.

1. **Environment setup**: Make sure you have the following packages installed in your environment:
   ```bash
   pip install opencv-python librosa moviepy pandas numpy scikit-learn tqdm
   ```

2. **Run Pipeline**:
   The primary execution runs through `reels_solution.ipynb`. It automatically reads unstructured video data from the `reels` directory and generates multimodal analytics.
   
3. **Artifacts**: 
   A successful execution will export the final clustered data signals to `submission.csv` containing metrics across all processed videos.

   
