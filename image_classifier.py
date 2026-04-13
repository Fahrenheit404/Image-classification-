#import libraries
import numpy as np
import tensorflow as tf
from keras import datasets, layers, models
from keras.utils import to_categorical
import random
import matplotlib.pyplot as plt


(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Normalize pixel values
train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshape for CNN (Samples, Height, Width, Channels)
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

# One-hot encode labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Train the model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

print("Training started...")
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_data=(test_images, test_labels))

# Evaluate
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"\nFinal Test Accuracy: {test_accuracy*100:.2f}%")

# visualize random prediction
idx = random.randint(0, len(test_images) - 1)
predictions = model.predict(test_images)
predicted_label = np.argmax(predictions[idx])
actual_label = np.argmax(test_labels[idx])



plt.imshow(test_images[idx].reshape(28,28), cmap='gray')
plt.title(f"Predicted: {predicted_label} | Actual: {actual_label}")
plt.axis('off')
plt.show()

print(f"Testing Index #{idx}: The model predicted {predicted_label}.")