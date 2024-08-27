import cv2
import torch
import torchvision.transforms as transforms

# Load the trained model
model = SimpleCNN(num_classes=num_classes)
model.load_state_dict(torch.load('emotion_model.pth', map_location=torch.device('cpu')))
model.eval()

# Define emotion labels
emotion_labels = image_datasets['train'].classes

# Capture webcam feed and detect emotion
def capture_emotion():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face / 255.0
        face = torch.tensor(face, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            output = model(face)
            _, predicted = torch.max(output, 1)
        
        emotion = emotion_labels[predicted.item()]
        return emotion

    cap.release()
    return None
