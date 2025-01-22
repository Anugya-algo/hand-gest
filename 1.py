import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import os
import cv2
import mediapipe as mp
import keras
import mediapipe_model_maker
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from pathlib import Path

class ImageAugmentor:
    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            brightness_range=[0.7, 1.3],
            zoom_range=0.2,
            fill_mode='nearest'
        )

    def load_image(self, image_path):
        """Load image and normalize"""
        img = tf.io.read_file(str(image_path))
        img = tf.image.decode_image(img, channels=3)
        img = tf.cast(img, tf.float32) / 255.0
        return img

    def save_image(self, image, output_path):
        """Save augmented image"""
        image = tf.clip_by_value(image, 0, 1) * 255
        image = tf.cast(image, tf.uint8)
        tf.io.write_file(str(output_path), tf.image.encode_jpeg(image))

    def apply_gaussian_blur(self, image):
        """Apply Gaussian blur"""
        return gaussian_blur(image, filter_size=5, sigma=1.0)

    def apply_noise(self, image):
        """Add Gaussian noise"""
        noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.1)
        return tf.clip_by_value(image + noise, 0.0, 1.0)

    def apply_edge_enhancement(self, image):
        """Apply edge enhancement using Sobel operator"""
        gray = tf.image.rgb_to_grayscale(image)
        sobel_x = tf.image.sobel_edges(tf.expand_dims(gray, 0))[:, :, :, :, 0]
        sobel_y = tf.image.sobel_edges(tf.expand_dims(gray, 0))[:, :, :, :, 1]
        edges = tf.sqrt(tf.square(sobel_x) + tf.square(sobel_y))[0]
        return tf.image.grayscale_to_rgb(edges)

    def apply_grayscale(self, image):
        """Convert to grayscale"""
        gray = tf.image.rgb_to_grayscale(image)
        return tf.image.grayscale_to_rgb(gray)

    def augment_images(self, num_augmentations=5):
        """Augment all images in the input directory"""
        for class_dir in self.input_dir.iterdir():
            if not class_dir.is_dir():
                continue

            # Create output class directory
            output_class_dir = self.output_dir / class_dir.name
            output_class_dir.mkdir(exist_ok=True)

            for img_path in class_dir.glob('*.[jJ][pP][gG]'):
                print(f"Processing {img_path}")

                # Load image
                image = self.load_image(img_path)

                # Basic augmentations using ImageDataGenerator
                img_array = tf.expand_dims(image, 0)
                aug_iterator = self.datagen.flow(img_array, batch_size=1)

                # Generate augmented images
                for i in range(num_augmentations):
                    # Get base augmented image
                    aug_image = aug_iterator.next()[0]

                    # Apply different augmentation combinations
                    augmentations = {
                        'basic': aug_image,
                        'blur': self.apply_gaussian_blur(aug_image),
                        'noise': self.apply_noise(aug_image),
                        'edges': self.apply_edge_enhancement(aug_image),
                        'gray': self.apply_grayscale(aug_image)
                    }

                    # Save each augmented version
                    for aug_name, aug_img in augmentations.items():
                        output_path = output_class_dir / f"{img_path.stem}_{aug_name}_{i}{img_path.suffix}"
                        self.save_image(aug_img, output_path)
    def augment_images(self, num_augmentations=5):
        """
        Augment all images in the input directory
        """
        # Process each class directory
        for class_dir in self.input_dir.iterdir():
            if not class_dir.is_dir():
                continue

            # Create output class directory
            output_class_dir = self.output_dir / class_dir.name
            output_class_dir.mkdir(exist_ok=True)

            # Process each image in the class directory
            for img_path in class_dir.glob('*.[jJ][pP][gG]'):
                print(f"Processing {img_path}")
                
                # Load image
                image = self.load_image(img_path)
                
                # Basic augmentations using ImageDataGenerator
                img_array = tf.expand_dims(image, 0)
                aug_iterator = self.datagen.flow(img_array, batch_size=1)
                
                # Generate augmented images
                for i in range(num_augmentations):
                    # Get base augmented image
                    aug_image = aug_iterator.next()[0]
                    
                    # Apply different augmentation combinations
                    augmentations = {
                        'basic': aug_image,
                        'blur': self.apply_gaussian_blur(aug_image),
                        'noise': self.apply_noise(aug_image),
                        'edges': self.apply_edge_enhancement(aug_image),
                        'gray': self.apply_grayscale(aug_image)
                    }
                    
                    # Save each augmented version
                    for aug_name, aug_img in augmentations.items():
                        output_path = output_class_dir / f"{img_path.stem}_{aug_name}_{i}{img_path.suffix}"
                        self.save_image(aug_img, output_path)

class GestureANNTrainer:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
        
    def extract_landmarks(self, image):
        """Extract hand landmarks using MediaPipe"""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image
        results = self.hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            landmarks = []
            # Get landmarks for first hand
            for landmark in results.multi_hand_landmarks[0].landmark:
                # Extract x, y, z coordinates for each landmark
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            return np.array(landmarks)
        return None

    def prepare_dataset(self, data_dir):
        """Process images and extract landmarks"""
        X = []  # Landmark vectors
        y = []  # Labels
        
        # Process each gesture class
        for gesture_class in tqdm(os.listdir(data_dir), desc="Processing gesture classes"):
            class_dir = os.path.join(data_dir, gesture_class)
            if not os.path.isdir(class_dir):
                continue
                
            # Process each image in class
            for img_file in tqdm(os.listdir(class_dir), desc=f"Processing {gesture_class} images"):
                if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                img_path = os.path.join(class_dir, img_file)
                image = cv2.imread(img_path)
                
                if image is None:
                    print(f"Failed to load image: {img_path}")
                    continue
                
                # Extract landmarks
                landmarks = self.extract_landmarks(image)
                
                if landmarks is not None:
                    X.append(landmarks)
                    y.append(gesture_class)
        
        return np.array(X), np.array(y)

    def build_model(self, input_shape, num_classes):
        """Build and compile the ANN model"""
        model = Sequential([
           
            Dense(256, input_shape=(input_shape,), activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
         
            Dense(68, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
          
            Dense(num_classes, activation='softmax')
        ])
        
      
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def train_model(self, data_dir, epochs=100, batch_size=32):
        """Train the ANN model"""
        # Prepare dataset
        print("Preparing dataset...")
        X, y = self.prepare_dataset(data_dir)
        
        if len(X) == 0:
            raise ValueError("No valid landmarks extracted from images")
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        y_onehot = tf.keras.utils.to_categorical(y_encoded)
        
        # # Split dataset
        # X_train, X_test, y_train, y_test = train_test_split(
        #     X, y_onehot, test_size=0.2, random_state=42
        # )
        
        # Build model
        num_landmarks = 21  # MediaPipe hand landmarks
        input_shape = num_landmarks * 3  # x, y, z coordinates for each landmark
        num_classes = len(label_encoder.classes_)
        
        model = self.build_model(input_shape, num_classes)
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_gesture_model.h5',
                monitor='val_accuracy',
                save_best_only=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5
            )
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        
        return model, label_encoder, history

def main():

    # Example usage
    input_directory = r"C:\Users\finda\Downloads\dataset_final\HandNavigation\Train"
    output_directory = r"C:\Users\finda\Downloads\dataset_final\HandNavigation\aug train"
    
    # Create augmentor
    augmentor = ImageAugmentor(input_directory, output_directory)
    
    # Perform augmentation
    augmentor.augment_images(num_augmentations=5)

    # Example usage
    trainer = GestureANNTrainer()
    
    # Train model
    model, label_encoder, history = trainer.train_model(
        data_dir='path/to/gesture/dataset',
        epochs=50,
        batch_size=32
    )
    
    # Save model and label encoder
    model.save('gesture_recognition_model.h5')
    np.save('label_encoder_classes.npy', label_encoder.classes_)

if __name__ == "__main__":
    main()