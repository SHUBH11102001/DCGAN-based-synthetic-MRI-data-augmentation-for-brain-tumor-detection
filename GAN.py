import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout, BatchNormalization
from keras.layers import Activation
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
import seaborn as sns
import warnings
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import shuffle


# Data paths
path_tumor = '/kaggle/input/brain-mri-images-for-brain-tumor-detection/yes'
path_normal = '/kaggle/input/brain-mri-images-for-brain-tumor-detection/no'

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Model parameters
IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS = 128, 128, 1
TOTAL_EPOCHS = 1000
NOISE_VECTOR_SIZE = 100
BATCH_AMOUNT = 64
RANDOM_SEED = 11

def load_images(directory, label):
    data, labels = [], []
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            image = cv2.imread(file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
            data.append(image)
            labels.append(label)
        except Exception as error:
            print(f"Error loading {filename}: {error}")
            continue
    
    return np.array(data), np.array(labels)

# Loading and processing images
tumor_data, tumor_labels = load_images(path_tumor, label=1)
normal_data, normal_labels = load_images(path_normal, label=0)

print(f"Tumor images loaded: {len(tumor_data)}")
print(f"Normal images loaded: {len(normal_data)}")


def display_sample_images(images, num_samples=15):
    plt.figure(figsize=(40, 40))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.axis('off')
    plt.show()

display_sample_images(tumor_data) 
display_sample_images(normal_data)    

def plot_pixel_intensity(images):
    pixel_values = images.flatten()
    plt.figure(figsize=(10, 6))
    sns.histplot(pixel_values, bins=50, kde=True)
    plt.title("Pixel Intensity Distribution")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.show()
    
plot_pixel_intensity(tumor_data)
plot_pixel_intensity(tumor_data)
        
training_data = tumor_data
print(training_data.shape) 

# Normalize and reshape
training_data = (training_data.astype(np.float32) - 127.5) / 127.5
training_data = training_data.reshape(-1, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
print(training_data.shape)   

optimizer = Adam(0.0002, 0.5)

# Generator Model
def create_generator():
    generator_model = Sequential([
        Dense(16*16*256, input_dim=NOISE_VECTOR_SIZE),
        LeakyReLU(alpha=0.2),
        Reshape((16, 16, 256)),    
        Conv2DTranspose(128, (4, 4), strides=2, padding='same'),
        BatchNormalization(momentum=0.8),
        LeakyReLU(alpha=0.2),        
        Conv2DTranspose(128, (4, 4), strides=2, padding='same'),
        BatchNormalization(momentum=0.8),
        LeakyReLU(alpha=0.2),
        Conv2DTranspose(128, (4, 4), strides=2, padding='same'),
        BatchNormalization(momentum=0.8),
        LeakyReLU(alpha=0.2),
        Conv2D(IMG_CHANNELS, (4, 4), padding='same', activation='tanh')
    ], name="gen_network")
    generator_model.compile(loss="binary_crossentropy", optimizer=optimizer)
    generator_model.summary()
    return generator_model


def create_discriminator():
    discriminator_model = Sequential([
        Conv2D(64, (3, 3), padding='same', input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)),
        LeakyReLU(alpha=0.2),
        Conv2D(128, (3, 3), strides=2, padding='same'),
        LeakyReLU(alpha=0.2),
        Conv2D(128, (3, 3), strides=2, padding='same'),
        LeakyReLU(alpha=0.2),        
        Conv2D(256, (3, 3), strides=2, padding='same'),
        LeakyReLU(alpha=0.2),        
        Flatten(),
        Dropout(0.4),
        Dense(1, activation="sigmoid")
    ], name="disc_network")
    discriminator_model.compile(loss="binary_crossentropy", optimizer=optimizer)    
    discriminator_model.summary()
    return discriminator_model

# Instantiating models
discriminator = create_discriminator()
generator = create_generator()
discriminator.trainable = False 
# GAN setup
input_layer = Input(shape=(NOISE_VECTOR_SIZE,))
fake_image = generator(input_layer)
output_layer = discriminator(fake_image)
gan_model = Model(input_layer, output_layer, name="combined_gan")
gan_model.compile(loss="binary_crossentropy", optimizer=optimizer)
print("Combined GAN Model:\n")
gan_model.summary()


def display_generated_images(noise_vector, plot_layout, fig_size=(22, 8), save_images=False):
    generated_imgs = generator.predict(noise_vector)
    plt.figure(figsize=fig_size)    
    for index, img in enumerate(generated_imgs):
        plt.subplot(plot_layout[0], plot_layout[1], index+1)
        if IMG_CHANNELS == 1:
            plt.imshow(img.reshape((IMG_WIDTH, IMG_HEIGHT)), cmap='gray')    
        else:
            plt.imshow(img.reshape((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)))
        if save_images:
            plt.savefig(f"generated_img_{index}")
        plt.axis('off')    
    plt.tight_layout()
    plt.show()
    
    
# GAN Training
np.random.seed(RANDOM_SEED)
num_batches_per_epoch = training_data.shape[0] // BATCH_AMOUNT
for epoch in range(TOTAL_EPOCHS):
    for step in tqdm(range(num_batches_per_epoch)):
        noise_vector = np.random.normal(0, 1, size=(BATCH_AMOUNT, NOISE_VECTOR_SIZE))
        generated_data = generator.predict(noise_vector)
        indices = np.random.randint(0, training_data.shape[0], size=BATCH_AMOUNT)
        real_data = training_data[indices]
        combined_data = np.concatenate((real_data, generated_data))
        discriminator_targets = np.zeros(2 * BATCH_AMOUNT)
        discriminator_targets[:BATCH_AMOUNT] = 1  

        # Train discriminator
        disc_loss = discriminator.train_on_batch(combined_data, discriminator_targets)
        
        # Train generator
        gen_targets = np.ones(BATCH_AMOUNT)  
        gen_loss = gan_model.train_on_batch(noise_vector, gen_targets)

    print(f"EPOCH: {epoch + 1} - Generator Loss: {gen_loss:.4f}, Discriminator Loss: {disc_loss:.4f}")
    if (epoch + 1) % 100 == 0:
        print(f"Displaying generated images at epoch {epoch + 1}")
        noise_vector = np.random.normal(0, 1, size=(10, NOISE_VECTOR_SIZE))
        display_generated_images(noise_vector, (2, 5))
    

def save_models(generator, discriminator, gan_model, model_name_prefix):
    generator.save(f"{model_name_prefix}_generator.h5")
    discriminator.save(f"{model_name_prefix}_discriminator.h5")
    gan_model.save(f"{model_name_prefix}_gan.h5")  
    
save_models(generator, discriminator, gan_model, model_name_prefix="tumor_GAN")

noise = np.random.normal(0, 1, size=(400, NOISE_VECTOR_SIZE))
tumor_generated_images = generator.predict(noise)
tumor_generated_images.shape


# TILL HERE WE HAVE GENERATED TUMOR IMAGES
# NOW WE WILL TRY TO GENERATE THE HEALTHY IMAGES WITHOUT TUMOR

training_data_normal = normal_data
training_data_normal = (training_data_normal.astype(np.float32) - 127.5) / 127.5
training_data_normal = training_data_normal.reshape(-1, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
training_data_normal.shape

generator_normal = create_generator()  
discriminator_normal = create_discriminator()  
discriminator_normal.trainable = False
gan_input_normal = Input(shape=(NOISE_VECTOR_SIZE,))
fake_image_normal = generator_normal(gan_input_normal)
gan_output_normal = discriminator_normal(fake_image_normal)
gan_normal = Model(gan_input_normal, gan_output_normal, name="gan_model_normal")
gan_normal.compile(loss="binary_crossentropy", optimizer=optimizer)
gan_normal.summary()

def display_generated_images_normal(noise_vector, plot_layout, fig_size=(22, 8), save_images=False):
    generated_imgs = generator_normal.predict(noise_vector)
    plt.figure(figsize=fig_size)
    
    for index, img in enumerate(generated_imgs):
        plt.subplot(plot_layout[0], plot_layout[1], index+1)
        if IMG_CHANNELS == 1:
            plt.imshow(img.reshape((IMG_WIDTH, IMG_HEIGHT)), cmap='gray')    
        else:
            plt.imshow(img.reshape((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)))
        if save_images:
            plt.savefig(f"generated_img_{index}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
# GAN Training
import numpy as np
from tqdm import tqdm

np.random.seed(RANDOM_SEED)
num_batches_per_epoch = training_data_normal.shape[0] // BATCH_AMOUNT

for epoch in range(TOTAL_EPOCHS):
    for step in tqdm(range(num_batches_per_epoch)):
        noise_vector = np.random.normal(0, 1, size=(BATCH_AMOUNT, NOISE_VECTOR_SIZE))
        generated_data = generator_normal.predict(noise_vector)        
        indices = np.random.randint(0, training_data_normal.shape[0], size=BATCH_AMOUNT)
        real_data = training_data_normal[indices]
        combined_data = np.concatenate((real_data, generated_data))
        discriminator_targets = np.zeros(2 * BATCH_AMOUNT)
        discriminator_targets[:BATCH_AMOUNT] = 1  

        # Train discriminator
        disc_loss = discriminator_normal.train_on_batch(combined_data, discriminator_targets)
        
        # Train generator
        gen_targets = np.ones(BATCH_AMOUNT)  
        gen_loss = gan_normal.train_on_batch(noise_vector, gen_targets)

    print(f"EPOCH: {epoch + 1} - Generator Loss: {gen_loss:.4f}, Discriminator Loss: {disc_loss:.4f}")

    if (epoch + 1) % 100 == 0:
        print(f"Displaying generated images at epoch {epoch + 1}")
        noise_vector = np.random.normal(0, 1, size=(10, NOISE_VECTOR_SIZE))
        display_generated_images_normal(noise_vector, (2, 5))    
    
save_models(generator_normal, discriminator_normal, gan_normal, model_name_prefix="normal_GAN")


normal_generated_images = generator_normal.predict(noise)



# LETS COMBINE THE GENERATED IMAGES(BOTH TUMOR AND NORMAL)
combined_generated_images = np.concatenate((tumor_generated_images, normal_generated_images), axis=0)     
tumor_labels_gen = np.ones(tumor_generated_images.shape[0])  # Array of 1s for tumor images
normal_labels_gen = np.zeros(normal_generated_images.shape[0])  # Array of 0s for normal images
combined_labels = np.concatenate((tumor_labels, normal_labels), axis=0)  

combined_generated_images, combined_labels = shuffle(
    combined_generated_images, combined_labels, random_state=42
)

# WE WILL USE THE ACTUAL DATASET AS THE VALIDATION DATA FOR OUR CLASSIFICATION MODELS
val_data = np.concatenate((tumor_data, normal_data), axis=0)
val_labels = np.concatenate((tumor_labels, normal_labels), axis=0)
val_data = np.expand_dims(val_data, axis=-1)  


train_gen = ImageDataGenerator(
    rescale=1./2.,
    rotation_range=15,        
    width_shift_range=0.1,    
    height_shift_range=0.1,    
    shear_range=0.1,          
    zoom_range=0.1,         
    horizontal_flip=True,      
    fill_mode='nearest'      
)

val_gen = ImageDataGenerator(rescale=1./255)

train_generator = train_gen.flow(
    combined_generated_images, 
    combined_labels, 
    batch_size=32,
    shuffle=True  
)

val_generator = val_gen.flow(
    val_data, 
    val_labels, 
    batch_size=32
)


# WE WILL NOW START WITH OUR FIRST CLASSIFICATION MODEL USING RESNET152V2
input_shape = (128, 128, 1)
def create_model(input_shape):
    base_model = ResNet152V2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    base_model.trainable = False
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))
    model.add(layers.Conv2D(3, (1, 1), padding='same', input_shape=input_shape))
    model.add(base_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))  

    return model

model_resnet = create_model(input_shape)
model_resnet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_resnet.summary()
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

history_resnet = model_resnet.fit(
    train_generator,              
    epochs=100,                    
    validation_data=val_generator, 
    callbacks=[early_stopping]    
)

plt.figure(figsize=(12, 6))
plt.plot(history_resnet.history['accuracy'], label='Training Accuracy')
plt.plot(history_resnet.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()


val_predictions = model_resnet.predict(val_data)
val_predictions = (val_predictions > 0.5).astype(int)
accuracy = accuracy_score(val_labels, val_predictions)
print(f"Classification Accuracy: {accuracy:.2f}")


# USING MOBILENETV2

def mobilenet_model(input_shape):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    base_model.trainable = False
    model = models.Sequential()    
    model.add(layers.InputLayer(input_shape=input_shape))
    model.add(layers.Conv2D(3, (1, 1), padding='same', input_shape=input_shape))    
    model.add(base_model)    
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(1, activation='sigmoid'))  
    return model

model_mobilenet = mobilenet_model(input_shape)
model_mobilenet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_mobilenet.summary()

early_stopping_mob = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

# Train the model
history_mobilenet = model_mobilenet.fit(
    train_generator,              
    epochs=100,                    
    validation_data=val_generator, 
    callbacks=[early_stopping_mob]    
)

plt.figure(figsize=(12, 6))
plt.plot(history_mobilenet.history['accuracy'], label='Training Accuracy')
plt.plot(history_mobilenet.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()

val_predictions = model_mobilenet.predict(val_data)
val_predictions = (val_predictions > 0.5).astype(int)
accuracy = accuracy_score(val_labels, val_predictions)
print(f"Classification Accuracy: {accuracy:.2f}")

