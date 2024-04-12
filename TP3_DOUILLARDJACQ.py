import tensorflow as tf
import matplotlib.pyplot as plt


mnist = tf.keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train / 255.0  # Normalize pixel values to be between 0 and 1
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1) # Additional dimension for color
Y_train = tf.keras.utils.to_categorical(Y_train)
print("Shape X_train", X_train.shape) # X_train is a 4D matrix

model = tf.keras.models.Sequential()

# Add Convolutional Layers
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))

# Add Pooling Layers (MaxPooling2D or AveragePooling2D)
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

# Flatten the Output
model.add(tf.keras.layers.Flatten())

# Add Fully Connected Layers
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))  # Dropout layer for regularization
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))  # Output layer with softmax activation

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Display the model summary
model.summary()

# Train the model (if not already trained)
model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, tf.keras.utils.to_categorical(Y_test))

# Print the evaluation results
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

model.save("Model.h5")

import pygame
import numpy as np

# Load the model
model = tf.keras.models.load_model("Model.h5")

pygame.init() # initialize Pygame
BLACK = (0, 0, 0) # defining black color
WHITE = (255, 255, 255) # defining black color

# Set up the display
WIDTH, HEIGHT = 400, 400
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Handwritten Digit Recognition")

window.fill(WHITE)

drawing = False  # To track whether the user is drawing
last_pos = None  # To store the previous mouse position for smooth lines
radius = 15 # To define the font size of the drawing tool

def predict_digit(img):
    img = tf.image.resize(img, [28, 28])
    img = tf.reshape(img, (1, 28, 28, 1))
    img = img / 255.0
    img = tf.image.transpose(img)
    img = 1.0 - img
    prediction = model.predict(img)
    print("Raw Prediction Scores:", prediction)
    digit = np.argmax(prediction)
    return digit

def display_text(text):
    font = pygame.font.Font(None, 72)
    text_surface = font.render(text, True, BLACK)
    text_rect = text_surface.get_rect(topright=(WIDTH - 20, 20))
    window.blit(text_surface, text_rect)

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True
            last_pos = pygame.mouse.get_pos()
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
        elif event.type == pygame.MOUSEMOTION:
            if drawing:
                pos = pygame.mouse.get_pos()
                if last_pos is not None:
                    pygame.draw.line(window, BLACK, last_pos, pos, radius)
                last_pos = pos
    keys = pygame.key.get_pressed()
    if keys[pygame.K_SPACE]:
        window.fill(WHITE)
    if keys[pygame.K_RETURN]:
        img = pygame.surfarray.array3d(window)
        img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
        img = np.expand_dims(img, axis=-1) # add channel dimension
        digit = predict_digit(img)
        display_text(str(digit))
    pygame.display.flip()
pygame.quit()
