import cv2
import numpy as np
import pytesseract
from moviepy import VideoFileClip
import speech_recognition as sr
import instaloader
import requests
import os
from transformers import pipeline
import librosa
from concurrent.futures import ThreadPoolExecutor
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
import json

class VideoLabeler:
    def __init__(self):
        # Load classifications from JSON
        with open('classifications.json', 'r') as f:
            self.classifications = json.load(f)
        
        # Initialize models and tools
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", revision="714eb0f")
        self.emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", revision="main")
        self.recognizer = sr.Recognizer()
        self.L = instaloader.Instaloader()
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.object_detector = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    def download_instagram_reel(self, url):
        # ... (unchanged)
        try:
            shortcode = url.split('/')[-2]
            post = instaloader.Post.from_shortcode(self.L.context, shortcode)
            video_url = post.video_url
            
            video_path = f"temp_video_{shortcode}.mp4"
            with open(video_path, 'wb') as f:
                f.write(requests.get(video_url).content)
            return video_path
        except Exception as e:
            print(f"Error downloading video: {e}")
            return None

    def extract_audio(self, video_path):
        # ... (unchanged)
        try:
            video = VideoFileClip(video_path)
            audio_path = video_path.replace('.mp4', '.wav')
            video.audio.write_audiofile(audio_path)
            video.close()
            return audio_path
        except Exception as e:
            print(f"Error extracting audio: {e}")
            return None

    def analyze_text_content(self, video_path):
        # ... (unchanged)
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        video = cv2.VideoCapture(video_path)
        frames_analyzed = 0
        text_content = []
        
        while video.isOpened() and frames_analyzed < 30:
            ret, frame = video.read()
            if not ret:
                break
            text = pytesseract.image_to_string(frame)
            if text.strip():
                text_content.append(text.strip())
            frames_analyzed += 1
            
        video.release()
        
        if text_content:
            sentiment = self.sentiment_analyzer(text_content)
            emotions = self.emotion_classifier(text_content)
            return {
                'text': text_content,
                'sentiment': sentiment,
                'emotions': emotions
            }
        return None

    def analyze_audio(self, audio_path):
        try:
            with sr.AudioFile(audio_path) as source:
                audio = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio)
                
                sentiment = self.sentiment_analyzer(text)
                emotions = self.emotion_classifier(text)
                
                y, sample_rate = librosa.load(audio_path)
                tempo, _ = librosa.beat.beat_track(y=y, sr=sample_rate)
                energy = librosa.feature.rms(y=y).mean()
                duration = librosa.get_duration(y=y, sr=sample_rate)
                
                sound_types = self.classify_sound_type(text, tempo, energy)
                audio_labels = self.generate_audio_labels(text, emotions, tempo, energy, duration)
                
                return {
                    'text': text,
                    'sentiment': sentiment,
                    'emotions': emotions,
                    'tempo': tempo,
                    'energy': energy,
                    'sound_types': sound_types,
                    'audio_labels': audio_labels,
                    'duration': duration
                }
        except Exception as e:
            print(f"Error analyzing audio: {e}")
            return None

    def classify_sound_type(self, text, tempo, energy):
        sound_types = set()
        for label, criteria in self.classifications['sound_types'].items():
            if 'tempo_min' in criteria and tempo > criteria['tempo_min'] and 'energy_min' in criteria and energy > criteria['energy_min']:
                sound_types.add(label)
            elif 'speech_confidence' in criteria and text.strip():
                sound_types.add(label)
            elif 'word_count_min' in criteria and len(text.split()) > criteria['word_count_min']:
                sound_types.add(label)
            elif 'energy_max' in criteria and energy < criteria['energy_max']:
                sound_types.add(label)
            elif 'speech_confidence_max' in criteria and not text.strip() and energy > criteria.get('energy_min', 0):
                sound_types.add(label)
        return sound_types

    def detect_objects(self, video_path):
        # ... (unchanged)
        video = cv2.VideoCapture(video_path)
        detected_objects = set()
        frames_analyzed = 0
        
        while video.isOpened() and frames_analyzed < 5:
            ret, frame = video.read()
            if not ret:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inputs = self.processor(images=frame_rgb, return_tensors="pt")
            outputs = self.object_detector(**inputs)
            
            target_sizes = torch.tensor([frame_rgb.shape[:2]])
            results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
            
            for label in results["labels"]:
                detected_objects.add(self.object_detector.config.id2label[label.item()])
                
            frames_analyzed += 1
            
        video.release()
        return detected_objects

    def analyze_video_content(self, video_path):
        # ... (unchanged)
        video = cv2.VideoCapture(video_path)
        frame_count = 0
        motion_score = 0
        prev_frame = None
        
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_frame is not None:
                diff = cv2.absdiff(prev_frame, gray)
                motion_score += np.mean(diff)
            
            prev_frame = gray
            frame_count += 1
            
        video.release()
        
        return {
            'motion_score': motion_score / frame_count if frame_count > 0 else 0,
            'frame_count': frame_count
        }

    def generate_text_labels(self, text_analysis):
        labels = set()
        if not text_analysis:
            return labels
            
        avg_sentiment = np.mean([s['score'] for s in text_analysis['sentiment']])
        emotions = [e['label'] for e in text_analysis['emotions']]
        text_content = ' '.join(text_analysis['text']).lower()
        
        for label, criteria in self.classifications['text_labels'].items():
            if 'sentiment_min' in criteria and avg_sentiment > criteria['sentiment_min']:
                if 'emotion_requirements' in criteria and any(e in emotions for e in criteria['emotion_requirements']):
                    labels.add(label)
                elif 'text_count_min' in criteria and len(text_analysis['text']) > criteria['text_count_min']:
                    labels.add(label)
            if 'keywords' in criteria and any(kw in text_content for kw in criteria['keywords']):
                labels.add(label)
                
        return labels

    def generate_audio_labels(self, text, emotions, tempo, energy, duration):
        labels = set()
        text_content = text.lower()
        
        for label, criteria in self.classifications['audio_labels'].items():
            if 'tempo_min' in criteria and tempo > criteria['tempo_min'] and 'energy_min' in criteria and energy > criteria['energy_min']:
                labels.add(label)
            if 'emotion_requirements' in criteria and any(e in emotions for e in criteria['emotion_requirements']):
                labels.add(label)
            if 'keywords' in criteria and any(kw in text_content for kw in criteria['keywords']):
                labels.add(label)
            if 'speech_confidence' in criteria and text.strip():
                labels.add(label)
            if 'word_count_min' in criteria and len(text.split()) > criteria['word_count_min']:
                labels.add(label)
            if 'energy_min' in criteria and 'tempo_max' in criteria and energy > criteria['energy_min'] and tempo < criteria['tempo_max']:
                labels.add(label)
            if 'duration_min' in criteria and duration > criteria['duration_min']:
                labels.add(label)
                
        return labels

    def generate_video_labels(self, video_analysis, text_labels, detected_objects):
        labels = set()
        
        for label, criteria in self.classifications['video_labels'].items():
            if 'motion_score_min' in criteria and video_analysis['motion_score'] > criteria['motion_score_min']:
                labels.add(label)
            if 'frame_count_max' in criteria and video_analysis['frame_count'] < criteria['frame_count_max']:
                if 'requires_funny' not in criteria or 'funny' in text_labels:
                    labels.add(label)
            if 'frame_count_min' in criteria and video_analysis['frame_count'] > criteria['frame_count_min']:
                labels.add(label)
            if 'object_requirements' in criteria and any(obj in detected_objects for obj in criteria['object_requirements']):
                labels.add(label)
                
        return labels

    def classify_reel_type(self, labels):
        reel_types = set()
        for reel_type, associated_labels in self.classifications['reel_types'].items():
            if any(label in labels for label in associated_labels):
                reel_types.add(reel_type)
        return reel_types

    def process_video(self, url):
        video_path = self.download_instagram_reel(url)
        if not video_path:
            return None

        with ThreadPoolExecutor() as executor:
            audio_future = executor.submit(self.extract_audio, video_path)
            text_future = executor.submit(self.analyze_text_content, video_path)
            video_future = executor.submit(self.analyze_video_content, video_path)
            object_future = executor.submit(self.detect_objects, video_path)

            audio_path = audio_future.result()
            text_analysis = text_future.result()
            video_analysis = video_future.result()
            detected_objects = object_future.result()

        audio_analysis = self.analyze_audio(audio_path) if audio_path else None
        
        text_labels = self.generate_text_labels(text_analysis)
        audio_labels = audio_analysis['audio_labels'] if audio_analysis else set()
        video_labels = self.generate_video_labels(video_analysis, text_labels, detected_objects)
        
        all_labels = text_labels.union(audio_labels, video_labels)
        reel_types = self.classify_reel_type(all_labels)

        # Clean up
        os.remove(video_path)
        if audio_path:
            os.remove(audio_path)

        return {
            'labels': all_labels,
            'sound_types': audio_analysis['sound_types'] if audio_analysis else set(),
            'reel_types': reel_types,
            'detected_objects': detected_objects,
            'text_labels': text_labels,
            'audio_labels': audio_labels,
            'video_labels': video_labels
        }

def main():
    labeler = VideoLabeler()
    url = input("Enter Instagram Reel URL: ")
    results = labeler.process_video(url)
    
    if results:
        print("All Labels:", ", ".join(results['labels']))
        print("Text Labels:", ", ".join(results['text_labels']))
        print("Audio Labels:", ", ".join(results['audio_labels']))
        print("Video Labels:", ", ".join(results['video_labels']))
        print("Sound Types:", ", ".join(results['sound_types']))
        print("Reel Types:", ", ".join(results['reel_types']))
        print("Detected Objects:", ", ".join(results['detected_objects']))
    else:
        print("Could not process video")

if __name__ == "__main__":
    main()