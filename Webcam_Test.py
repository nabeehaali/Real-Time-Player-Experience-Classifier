import cv2
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model('CNN_Emotion_Model.h5')

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to preprocess the image
def preprocess_image(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None, None
    (x, y, w, h) = faces[0]
    # Crop and resize the face region to 48x48
    face_roi = gray[y:y+w, x:x+h]
    face_resized = cv2.resize(face_roi, (48, 48))
    # Normalize pixel values
    face_normalized = face_resized / 255.0
    # Reshape for model input
    face_input = np.expand_dims(face_normalized, axis=0)
    face_input = np.expand_dims(face_input, axis=-1)
    return face_input, (x, y, w, h)

# Function to draw bounding box and predicted emotion on the image
def draw_emotion(image, bounding_box, emotion):
    (x, y, w, h) = bounding_box
    # Draw bounding box
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # Display predicted emotion
    cv2.putText(image, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess the captured image
    face_input, bounding_box = preprocess_image(frame)
    
    if face_input is not None:
        # Predict emotion
        emotion_index = np.argmax(model.predict(face_input))
        emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        emotion = emotions[emotion_index]
        
        # Draw bounding box and predicted emotion
        draw_emotion(frame, bounding_box, emotion)

    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
