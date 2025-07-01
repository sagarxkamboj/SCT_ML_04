import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

def load_data(dataset_path='leapGestRecog', img_size=(64, 64), max_images_per_class=1000):
    X = []
    y = []
    label_map = {}  # gesture_name -> numeric label
    label_index = 0

    print("📥 Loading dataset...")

    # Loop through all person folders (00 to 09)
    for person_folder in sorted(os.listdir(dataset_path)):
        person_path = os.path.join(dataset_path, person_folder)
        if not os.path.isdir(person_path):
            continue

        # Loop through each gesture folder inside person folder
        for gesture_folder in sorted(os.listdir(person_path)):
            gesture_path = os.path.join(person_path, gesture_folder)
            if not os.path.isdir(gesture_path):
                continue

            # Assign label number to gesture
            if gesture_folder not in label_map:
                label_map[gesture_folder] = label_index
                label_index += 1

            label = label_map[gesture_folder]
            image_count = 0

            for file in os.listdir(gesture_path):
                if not file.endswith('.png'):
                    continue
                if image_count >= max_images_per_class:
                    break

                file_path = os.path.join(gesture_path, file)
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue

                image = cv2.resize(image, img_size)
                X.append(image)
                y.append(label)
                image_count += 1

    if len(X) == 0:
        print("❌ No images loaded. Check the dataset path or format.")
        return [], [], {}

    X = np.array(X).reshape(-1, img_size[0], img_size[1], 1) / 255.0
    y = to_categorical(np.array(y), num_classes=len(label_map))

    print(f"✅ Loaded {len(X)} images across {len(label_map)} classes.")
    return X, y, label_map

def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --- Main code execution ---
if __name__ == "__main__":
    X, y, label_map = load_data(dataset_path='leapGestRecog', img_size=(64, 64), max_images_per_class=500)

    if len(X) == 0:
        exit()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and train model
    model = build_model(input_shape=X.shape[1:], num_classes=y.shape[1])
    model.summary()
    print("🚀 Training model...")
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

    # Evaluate model
    print("📊 Evaluating model...")
    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)

    print("✅ Accuracy:", accuracy_score(y_true_labels, y_pred_labels))
    print("\n📋 Classification Report:\n", classification_report(y_true_labels, y_pred_labels, target_names=label_map.keys()))

    # Save model
    model.save("gesture_cnn_model.h5")
    print("💾 Model saved as gesture_cnn_model.h5")
