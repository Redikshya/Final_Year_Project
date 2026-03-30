import pickle
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load preprocessed data
with open('data.pkl', 'rb') as f:
    x_train, x_test, y_train, y_test, label_encoder = pickle.load(f)

# Print shapes to verify the data
print(f"x_train shape: {x_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Defining the CNN Model Architecture
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping # Import EarlyStopping

def create_cnn_model(input_shape, num_classes):
    model = models.Sequential()

    # --- REGULARIZATION CHANGES START HERE ---

    # L2 Regularization added to Conv2D layers
    # kernel_regularizer=regularizers.l2(0.0001) penalizes large weights, simplifying the model.
    l2_reg = regularizers.l2(0.0001)

    # Convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=l2_reg))
    model.add(layers.MaxPooling2D((2, 2)))
    # Dropout layer added after pooling to randomly 'turn off' neurons, forcing the network to learn robust features.
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2_reg))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2_reg))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2_reg))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten and fully connected layers
    model.add(layers.Flatten())
    
    # Dropout added before the final Dense layer
    model.add(layers.Dropout(0.5)) 
    
    model.add(layers.Dense(128, activation='relu', kernel_regularizer=l2_reg))
    model.add(layers.Dense(num_classes, activation='softmax'))

    # --- REGULARIZATION CHANGES END HERE ---

    return model

# Define input shape and number of classes
input_shape = x_train.shape[1:]  # (height, width, channels)
num_classes = len(label_encoder.classes_)
model = create_cnn_model(input_shape, num_classes)

# Compiling the Model

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# --- EARLY STOPPING IMPLEMENTATION ---

# Early Stopping stops training when the monitored metric (val_loss) stops improving.
# monitor='val_loss': Watch the validation loss
# patience=10: Allow 10 epochs with no improvement before stopping
# restore_best_weights=True: Keep the model weights from the epoch with the lowest val_loss
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
callbacks = [early_stopping]

from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=10,      # rotate images by up to 10 degrees
    horizontal_flip=True,   # randomly flip images horizontally
)

# Fit the generator (required if featurewise normalization is used — harmless otherwise)
datagen.fit(x_train)
# Training the Model
epochs = 50 # Set a high number, but Early Stopping will likely stop it sooner
batch_size = 32

history = model.fit(
    datagen.flow(x_train, y_train, batch_size=batch_size),
    validation_data=(x_test, y_test),
    epochs=epochs,
    callbacks=callbacks
)# Pass the callbacks list here

# Saving the Model
# Save the trained model
model.save('face_recognition_model.h5')

# Save the LabelEncoder separately to decode predictions during recognition
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("Model and LabelEncoder saved.")
model.summary()

# 1. Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest Accuracy: {test_accuracy:.4f}")

# 2. Generate predictions and confusion matrix
y_pred = np.argmax(model.predict(x_test), axis=1)  # Get predicted classes
cm = confusion_matrix(y_test, y_pred)  # Confusion matrix

# 3. Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# 4. Print Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# 5. Plot Training and Validation Accuracy and Loss
def plot_training_history(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

# Call the function to plot training history
plot_training_history(history)