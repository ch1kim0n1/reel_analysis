# Instagram Reel Analyzer

## Overview

Instagram Reel Analyzer is a Python-based tool that performs comprehensive analysis of Instagram Reels. It uses computer vision, audio processing, and natural language processing to categorize video content, identify audio types, and detect objects within the video.

## Features

1. **Content Labeling**
   - Multi-label classification with an extensive library of Instagram-specific categories
   - Categories include: funny, meme, informational, educational, emotional, music, dance, vlog, tutorial, reaction, challenge, fitness, cooking, travel, fashion, beauty, gaming, review, unboxing, prank, storytime

2. **Audio Analysis**
   - Detects different types of audio: music, voiceover, informational, ambient, sound_effects
   - Analyzes tempo, energy, and speech content

3. **Object Detection**
   - Identifies objects in videos using YOLOv8
   - Categories include: human, animal, cartoon, object, food, vehicle, nature

4. **Technical Analysis**
   - Motion analysis
   - Sentiment and emotion analysis of text and audio
   - Text extraction from video frames

## Requirements

- Python 3.8+
- Dependencies:
  ```bash
  pip install opencv-python numpy pytesseract moviepy speechrecognition instaloader requests transformers librosa ultralytics


## This enhanced version includes:
1. Audio type classification
2. Extended content categories
3. Object detection using YOLOv8
4. Multiple label support with a comprehensive Instagram-specific library

To use this, you'll need to:
1. Install the additional dependency: `pip install ultralytics`
2. Update your requirements.txt file with all dependencies
3. Ensure you have sufficient memory as the object detection model requires more resources
