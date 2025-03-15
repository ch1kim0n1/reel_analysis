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

class VideoLabeler:
    def __init__(self):
        # Initialize models and tools
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
        self.recognizer = sr.Recognizer()
        self.L = instaloader.Instaloader()
        
        # Define label categories and thresholds
        self.label_thresholds = {
            'funny': 0.7,
            'meme': 0.6,
            'informational': 0.8,
            'educational': 0.8,
            'emotional': 0.7,
            'music': 0.6,
            'dance': 0.7
        }

    def download_instagram_reel(self, url):
        """Download Instagram Reel from URL"""
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
        """Extract audio from video"""
        try:
            video = VideoFileClip(video_path)
            audio_path = video_path.replace('.mp4', '.wav')
            video.audio.write_audiofile(audio_path)
            return audio_path
        except Exception as e:
            print(f"Error extracting audio: {e}")
            return None

    def analyze_text_content(self, video_path):
        """Extract and analyze text from video frames"""
        # Explicitly set the path to Tesseract executable
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        video = cv2.VideoCapture(video_path)
        frames_analyzed = 0
        text_content = []
        
        while video.isOpened() and frames_analyzed < 30:  # Analyze first 30 frames
            ret, frame = video.read()
            if not ret:
                break
                
            # Extract text from frame
            text = pytesseract.image_to_string(frame)
            if text.strip():
                text_content.append(text.strip())
            frames_analyzed += 1
            
        video.release()
        
        # Analyze extracted text
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
        """Analyze audio content"""
        try:
            with sr.AudioFile(audio_path) as source:
                audio = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio)
                
                # Analyze speech
                sentiment = self.sentiment_analyzer(text)
                emotions = self.emotion_classifier(text)
                
                # Analyze audio features
                y, sr = librosa.load(audio_path)
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                energy = librosa.feature.rms(y=y).mean()
                
                return {
                    'text': text,
                    'sentiment': sentiment,
                    'emotions': emotions,
                    'tempo': tempo,
                    'energy': energy
                }
        except Exception as e:
            print(f"Error analyzing audio: {e}")
            return None

    def analyze_video_content(self, video_path):
        """Analyze video motion and content"""
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
                # Calculate motion between frames
                diff = cv2.absdiff(prev_frame, gray)
                motion_score += np.mean(diff)
            
            prev_frame = gray
            frame_count += 1
            
        video.release()
        
        return {
            'motion_score': motion_score / frame_count if frame_count > 0 else 0,
            'frame_count': frame_count
        }

    def generate_labels(self, text_analysis, audio_analysis, video_analysis):
        """Generate consistent labels based on analysis"""
        labels = set()
        
        # Text-based labels
        if text_analysis:
            avg_sentiment = np.mean([s['score'] for s in text_analysis['sentiment']])
            emotions = [e['label'] for e in text_analysis['emotions']]
            
            if 'joy' in emotions and avg_sentiment > self.label_thresholds['funny']:
                labels.add('funny')
            if len(text_analysis['text']) > 2 and avg_sentiment > self.label_thresholds['informational']:
                labels.add('informational')
            if 'education' in ' '.join(text_analysis['text']).lower():
                labels.add('educational')

        # Audio-based labels
        if audio_analysis:
            if audio_analysis['tempo'] > 100 and audio_analysis['energy'] > 0.1:
                labels.add('music')
            emotions = [e['label'] for e in audio_analysis['emotions']]
            if 'joy' in emotions or 'anger' in emotions:
                labels.add('emotional')

        # Video-based labels
        if video_analysis:
            if video_analysis['motion_score'] > 50:
                labels.add('dance')
            if video_analysis['frame_count'] < 60 and 'funny' in labels:
                labels.add('meme')

        return labels

    def process_video(self, url):
        """Main processing function"""
        # Download video
        video_path = self.download_instagram_reel(url)
        if not video_path:
            return None

        # Process in parallel
        with ThreadPoolExecutor() as executor:
            audio_future = executor.submit(self.extract_audio, video_path)
            text_future = executor.submit(self.analyze_text_content, video_path)
            video_future = executor.submit(self.analyze_video_content, video_path)

            audio_path = audio_future.result()
            text_analysis = text_future.result()
            video_analysis = video_future.result()

        audio_analysis = self.analyze_audio(audio_path) if audio_path else None

        # Generate labels
        labels = self.generate_labels(text_analysis, audio_analysis, video_analysis)

        # Clean up temporary files
        os.remove(video_path)
        if audio_path:
            os.remove(audio_path)

        return labels

def main():
    labeler = VideoLabeler()
    url = input("Enter Instagram Reel URL: ")
    labels = labeler.process_video(url)
    
    if labels:
        print("Generated Labels:", ", ".join(labels))
    else:
        print("Could not process video")

if __name__ == "__main__":
    main()