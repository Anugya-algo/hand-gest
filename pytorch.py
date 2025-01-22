import os
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from sklearn.model_selection import train_test_split
from speech_recognition import Recognizer, Microphone
from PIL import Image
from torch.utils.data import Dataset, DataLoader


# Function to load dataset
def load_images_and_labels(dataset_dir, image_size=(224, 224)):
    images = []
    labels = []
    class_mapping = {}  # Map class names to numeric labels
    class_id = 0
    
    for class_name in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, class_name)
        if os.path.isdir(class_path):
            if class_name not in class_mapping:
                class_mapping[class_name] = class_id
                class_id += 1
            
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize(image_size)
                    images.append(np.array(img))
                    labels.append(class_mapping[class_name])
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
    
    return np.array(images), np.array(labels), class_mapping

# PyTorch Dataset class
class GestureDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Define your dataset directory
dataset_dir = r"C:\Users\finda\Downloads\dataset_final\HandNavigation\Train"

# Load images and labels
images, labels, class_mapping = load_images_and_labels(dataset_dir)

# Normalize and split dataset
images = images / 255.0  # Normalize images
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Create PyTorch Datasets
train_dataset = GestureDataset(X_train, y_train, transform=torch.tensor)
test_dataset = GestureDataset(X_test, y_test, transform=torch.tensor)

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Training loop placeholder
for epoch in range(10):  # Example number of epochs
    for images, labels in train_loader:
        # Move to device (GPU/CPU)
        images = images.float()  # Convert to float for PyTorch
        labels = labels.long()
        
        # Train your model here
        pass

print("Dataset loaded and ready for training!")


# 1. Data Augmentation
def augment_data(image):
    augmented_images = []
    transformations = [
        transforms.GaussianBlur(3),
        transforms.RandomGrayscale(p=1.0),
        transforms.ColorJitter(brightness=0.5),
    ]
    for transform in transformations:
        augmented_images.append(transform(image))
    return augmented_images


def extract_landmarks(image):
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        result = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                return [ (landmark.x, landmark.y, landmark.z) for landmark in hand_landmarks.landmark ]
    return None

# 3. Convert to Vector
def convert_to_vector(images, labels):
    vectors, vector_labels = [], []
    for img, label in zip(images, labels):
        landmarks = extract_landmarks(img)
        if landmarks:
            vectors.append(landmarks)
            vector_labels.append(label)
    return np.array(vectors), np.array(vector_labels)

# 4. Define ANN Model
class GestureModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(GestureModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], output_size)
        )
    
    def forward(self, x):
        return self.layers(x)

# 5. Train Model
def train_model(model, train_data, train_labels, epochs=10, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        inputs = torch.tensor(train_data, dtype=torch.float32)
        targets = torch.tensor(train_labels, dtype=torch.long)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# 6. Save and Load Model
def save_model(model, path="gesture_model.pth"):
    torch.save(model.state_dict(), path)

def load_model(path="gesture_model.pth"):
    model = GestureModel(input_size=63, hidden_sizes=[128, 64], output_size=10)
    model.load_state_dict(torch.load(path))
    return model

# 7. Real-Time Gesture Recognition
def predict_gesture(model, frame):
    landmarks = extract_landmarks(frame)
    if landmarks:
        input_vector = torch.tensor([landmarks], dtype=torch.float32)
        output = model(input_vector)
        _, predicted = torch.max(output, 1)
        return predicted.item()
    return None

# 8. Voice Recognition
def voice_command():
    recognizer = Recognizer()
    with Microphone() as source:
        print("Listening for command...")
        audio = recognizer.listen(source)
        try:
            return recognizer.recognize_google(audio)
        except:
            return "Error"

# Integration Example
if __name__ == "__main__":
    # Load dataset, augment, and split
    images, labels = convert_to_vector(images, labels)  # Define this based on your dataset
    augmented_images = [augment_data(img) for img in images]
    vectors, vector_labels = convert_to_vector(augmented_images, labels)
    X_train, X_test, y_train, y_test = train_test_split(vectors, vector_labels, test_size=0.2)

    # Train and save the model
    model = GestureModel(input_size=63, hidden_sizes=[128, 64], output_size=10)
    train_model(model, X_train, y_train)
    save_model(model)

    # Real-time prediction
    model = load_model()
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gesture = predict_gesture(model, frame)
        if gesture:
            print(f"Gesture: {gesture}")
        command = voice_command()
        if command:
            print(f"Voice Command: {command}")
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
