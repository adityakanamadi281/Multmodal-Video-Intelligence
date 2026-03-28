# Sparkonomy Unstructured Video Intelligence Challenge

## 📌 Executive Summary
This documentation outlines the methodology applied to the Sparkonomy ML Hacker Challenge. The primary objective is solving an unstructured, unannotated video problem by constructing a programmatic, scalable pipeline. 

Rather than defaulting to heavy deep learning frameworks, this solution leans heavily on **interpretable multimodal heuristics** (visual structure + acoustic characteristics) to build structured schemas, classify video archetypes, and calculate commercial applicability.

---

## 🏗️ Technical Approach: Multimodal Heuristics over Black Boxes
When exposed to zero-shot, unannotated environments, defaulting to large neural networks (like CNNs or Audio Transformers) often introduces unnecessary complexity, severe computing costs, and "black-box" outputs. 

To align perfectly with the challenge's emphasis on **practicality**, **interpretability**, and **speed**, this solution extracts direct physical metrics via **OpenCV** (Visual) and **Librosa/MoviePy** (Audio) using aggressive frame-skipping algorithms. 

### Why This Approach Wins:
- **Speed & Efficiency**: By sampling 1 out of every 10 frames (`sample_rate=10`), the visual extractor evaluates structural context almost instantly compared to continuous per-frame hardware inference.
- **Defensibility**: Every floating-point number outputted to `submission.csv` is mathematically grounded. We don't just output a vague probability that a video is an Ad; we confidently know it behaves like an Ad because it averages 1.5 scene cuts per second with an audio energy exceeding standard thresholds.

---

## 📊 The Signal Schema (What we extracted and why)

### 1. Visual Signatures
* **`editing_intensity`**: Calculated by detecting massive pixel variance spikes (`cv2.absdiff`) across sequential frames. 
  * **Why?** It strongly differentiates heavily edited promos and TikTok-style edits from continuous, single-take vlogs or podcasts.
* **`visual_quality`**: A synthetic composite derived from Contrast (pixel StdDev) and Brightness (pixel Mean). 
  * **Why?** It acts as an immediate proxy for production value (Professional studio lighting vs. amateur low-light webcam).
* **`max_faces`**: Utilizes lightweight Haar Cascades. 
  * **Why?** Instantly bifurcates Human-driven content (Vlog/Interview) from Product/Ambient B-Roll content.
* **`text_overlay_proxy`**: Calculates edge-density via Canny Edge detection specifically constrained to the lower third of the frame. 
  * **Why?** High sharp-edge density in this region strongly correlates with text subtitles, brand logos, and pop-ups common in commercial content.

### 2. Audio Signatures
* **`has_audio`**: Binary presence check.
* **`audio_energy`**: Mean Root Mean Square (RMS) amplitude of the audio waveform. 
  * **Why?** Loudness closely ties into high-energy marketing clips or intense creator engagement.
* **`speech_proxy_zcr`**: The global Zero-Crossing Rate. 
  * **Why?** Spoken consonants and percussive energetic beats cross the zero-axis thousands of times more accurately than flat ambient background music, providing extremely cheap separation of human speech vs. background noise.

### 3. Synthesized Commercial Intent
By mathematically blending the independent modalities above, the pipeline derives a custom **`commercial_intent_score`**. Videos demonstrating intense structural cuts, dense text overlays, and loud audio volumes are synthetically flagged as `High-Energy Ad/Promo`.

---

## 🚀 Scalability & Generalization (The "Bonus" Implementation)

During the interview, rely on this structured reasoning to explain how the local execution scales to production:

### Handling 1M+ Videos (Production Architecture)
1. **Serverless Orchestration**: Native raw video storage in Amazon S3/GCS can trigger asynchronous, serverless workers (AWS Lambda / Cloud Run) upon `ObjectCreated` events.
2. **The "Tiered Cascade" Resource Saver**: Inference on deep-learning models is financially expensive. This heuristic pipeline would be permanently deployed as the **Tier 1 Filter**. We only route videos that score highly on our cheap `commercial_intent_score` up towards heavy Tier 2 GPU instances (e.g., passing the video to OpenAI Whisper for semantic transcription). This drastically slashes cloud compute budgets by filtering out useless, silent, or unmarketable videos early.

### Zero-Shot Generalization
Because all signals are rooted entirely in universal physics (geometric edges, acoustic waveforms, color variance), the pipeline is **100% language, genre, and geography agnostic**. It performs exactly the same on a Japanese game show clip as it does on an American sports highlight without any retraining required.

---

## 💻 Execution
The entire pipeline is consolidated efficiently within the native Kaggle environment in `reels_solution.ipynb`. It programmatically walks the input directory, processes multimodal data, executes unsupervised topological clustering, and systematically dumps intelligence into `submission.csv`.
