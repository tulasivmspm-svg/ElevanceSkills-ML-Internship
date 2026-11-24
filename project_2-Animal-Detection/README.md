
# AI-Based Animal Detection System

An AI-powered wildlife detection and species classification application using YOLOv8 and Google SpeciesNet.

## Live Web Application

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://animal-detection-y4xnkqynxffyww5zrhedwh.streamlit.app/)

**Try the application:** https://animal-detection-y4xnkqynxffyww5zrhedwh.streamlit.app/

## Features

- Animal detection in images and videos
- Species-level classification using Google SpeciesNet
- Multiple animal detection
- Carnivore identification with red bounding boxes
- Herbivore identification with green bounding boxes
- Unknown animal identification
- Carnivore count and detection summary

## Technologies Used

- Python
- YOLOv8
- Google SpeciesNet
- OpenCV
- Streamlit
- PyTorch

## Model Evaluation

- Precision: 80.68%
- Recall: 79.54%
- mAP50: 84.94%
- mAP50-95: 69.12%

## Run Locally

```bash
pip install -r requirements.txt
streamlit run src/app.py
```
