import pyaudio
import numpy as np
import tensorflow as tf
import librosa
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
import cv2

# Constants for audio recording
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22050
CHUNK = 1024

# Load pre-trained speech emotion recognition model
emotion_model_audio = load_model('modelspeech.h5')

# Load pre-trained face emotion recognition model
emotion_model = load_model('model_face.h5')

# Initialize PyAudio for audio input
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# Initialize the webcam
cap = cv2.VideoCapture(0)

print("Detecting Emotions...")

while True:
    # Read microphone audio data
    data = stream.read(CHUNK)
    audio_data = np.frombuffer(data, dtype=np.int16)
    
    # Extract features from the audio data using librosa
    mfccs = librosa.feature.mfcc(y=audio_data.astype(float), sr=RATE, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    
    # Reshape the features for speech emotion prediction
    mfccs_processed = np.pad(mfccs_processed, ((0, 40 - mfccs_processed.shape[0])), mode='constant')
    audio_features = np.expand_dims(mfccs_processed, axis=0)
    audio_features = np.expand_dims(audio_features, axis=-1)
    
    # Normalize audio features
    audio_features = (audio_features - np.mean(audio_features)) / np.std(audio_features)
    
    # Predict speech emotion
    audio_prediction = emotion_model_audio.predict(audio_features)
    audio_emotion_label = ['Neutral','Calm' ,'Happy', 'Sad', 'Angry', 'Fearful','Disgust', 'Surprise'][np.argmax(audio_prediction)]
    
    
    
    emotion_dict = {0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}
    
    ret,frame = cap.read()
    frame = cv2.resize(frame,(1280,720))
    if not ret:
        break
    face_detector = cv2.CascadeClassifier('haarcascades_frontalface_default.xml') 
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    num_faces = face_detector.detectMultiScale(gray_frame,scaleFactor=1.3,minNeighbors = 5)
    
    for (x,y,w,h) in num_faces:
        cv2.rectangle(frame,(x,y-50),(x+w,y+h+10),(0,255,0),4)
        roi_gray_frame = gray_frame[y:y+h,x:x+w]
        cropped_img  = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame,(48,48)),-1),0)
        
        
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame,emotion_dict[maxindex],(x+5,y-20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,0)
        
    cv2.putText(frame, f"Speech Emotion: {audio_emotion_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('Combined Emotion Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and audio stream
cap.release()
cv2.destroyAllWindows()
stream.stop_stream()
stream.close()
audio.terminate()