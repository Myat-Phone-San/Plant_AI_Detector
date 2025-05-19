# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import cv2
# # import os
# # import random
# # from os import listdir
# # from PIL import Image
# # import tensorflow as tf
# # from keras.preprocessing.image import img_to_array
# # from keras.models import Sequential
# # from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
# # from sklearn.model_selection import train_test_split
# # from keras.utils import to_categorical
# # from keras.optimizers import Adam

# # # Set paths
# # dataset_path = r"C:\Users\lenovo\Documents\Plant-Disease-Detection-main\Dataset"
# # model_save_path = r"C:\Users\lenovo\Documents\Plant-Disease-Detection-main\Model\plant_disease_model.h5"

# # # Preview 16 random images from one class
# # def preview_images():
# #     class_path = os.path.join(dataset_path, "Potato___Early_blight")
# #     plt.figure(figsize=(12, 12))
# #     for i in range(1, 17):
# #         plt.subplot(4, 4, i)
# #         plt.tight_layout()
# #         rand_img = cv2.imread(os.path.join(class_path, random.choice(os.listdir(class_path))))
# #         rand_img = cv2.cvtColor(rand_img, cv2.COLOR_BGR2RGB)
# #         plt.imshow(rand_img)
# #         plt.title(f"{rand_img.shape[1]} x {rand_img.shape[0]}")
# #         plt.axis('off')
# #     plt.show()

# # # Convert image to array
# # def convert_image_to_array(image_path):
# #     try:
# #         image = cv2.imread(image_path)
# #         if image is not None:
# #             image = cv2.resize(image, (256, 256))
# #             return img_to_array(image)
# #         else:
# #             return np.array([])
# #     except Exception as e:
# #         print(f"Error: {e}")
# #         return None

# # def load_dataset():
# #     all_labels = ['Corn___Common_rust', 'Grape___Black_rot', 'Rice___Bacterial_leaf']
# #     image_list = []
# #     label_list = []

# #     for i, label_name in enumerate(all_labels):
# #         class_dir = os.path.join(dataset_path, label_name)
# #         print(f"Loading class: {label_name}")
# #         if not os.path.isdir(class_dir):
# #             print(f"Warning: {class_dir} does not exist!")
# #             continue

# #         for img_file in os.listdir(class_dir):
# #             img_path = os.path.join(class_dir, img_file)
# #             img_array = convert_image_to_array(img_path)
# #             if img_array.size != 0:
# #                 image_list.append(img_array)
# #                 label_list.append(i)  # Use index of label as class

# #     return np.array(image_list), np.array(label_list)


# # # def load_dataset():
# # #     root_dir = sorted(os.listdir(dataset_path))  # Ensures consistent ordering
# # #     image_list = []
# # #     label_list = []
# # #     all_labels = ['Corn___Common_rust', 'Grape___Black_rot', 'Rice___Bacterial_leaf']
    
# # #     binary_labels = list(range(len(root_dir)))


# # #     for i, directory in enumerate(root_dir):
# # #         print(f"Loading {directory}...")
# # #         plant_images = listdir(os.path.join(dataset_path, directory))
# # #         for img_file in plant_images:
# # #             img_path = os.path.join(dataset_path, directory, img_file)
# # #             image_list.append(convert_image_to_array(img_path))
# # #             label_list.append(binary_labels[i])
# # #     return np.array(image_list), np.array(label_list)

# # def build_model():
# #     model = Sequential()
# #     model.add(Conv2D(32, (3, 3), padding="same", input_shape=(256, 256, 3), activation="relu"))
# #     model.add(MaxPooling2D(pool_size=(3, 3)))
# #     model.add(Conv2D(16, (3, 3), padding="same", activation="relu"))
# #     model.add(MaxPooling2D(pool_size=(2, 2)))
# #     model.add(Flatten())
# #     model.add(Dense(8, activation="relu"))
# #     model.add(Dense(3, activation="softmax"))
# #     model.compile(loss='categorical_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])
# #     model.summary()
# #     return model

# # def main():
# #     # preview_images()  # Optional: Uncomment to preview dataset

# #     X, y = load_dataset()
# #     print(f"Dataset loaded: {X.shape[0]} images")

# #     # Normalize and reshape
# #     X = X.astype('float16') / 255.0
# #     X = X.reshape(-1, 256, 256, 3)
# #     y = to_categorical(y)

# #     # Train/test/validation split
# #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
# #     X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=10)

# #     model = build_model()

# #     # Train model
# #     history = model.fit(X_train, y_train, batch_size=128, epochs=50, validation_data=(X_val, y_val))

# #     # Save model
# #     model.save(model_save_path)
# #     print(f"Model saved to {model_save_path}")

# # if __name__ == "__main__":
# #     main()
# import numpy as np
# import os
# import random
# import cv2
# import matplotlib.pyplot as plt
# from keras.preprocessing.image import img_to_array
# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# from keras.utils import to_categorical
# from keras.optimizers import Adam
# from sklearn.model_selection import train_test_split

# # Set paths
# dataset_path = r"C:\Users\lenovo\Documents\Plant-Disease-Detection-main\Dataset"
# model_save_path = r"C:\Users\lenovo\Documents\Plant-Disease-Detection-main\Model\plant_disease_model.h5"

# # List of class names (with triple underscores)
# all_labels = ['Corn___Common_rust', 'Grape___Black_rot', 'Rice___Bacterialblight']

# # Preview 16 random images from one class
# def preview_images():
#     class_path = os.path.join(dataset_path, "Potato___Early_blight")
#     if not os.path.exists(class_path):
#         print(f"Folder not found: {class_path}")
#         return
#     plt.figure(figsize=(12, 12))
#     for i in range(16):
#         plt.subplot(4, 4, i + 1)
#         img_file = random.choice(os.listdir(class_path))
#         img = cv2.imread(os.path.join(class_path, img_file))
#         if img is None:
#             continue
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         plt.imshow(img)
#         plt.title(f"{img.shape[1]}x{img.shape[0]}")
#         plt.axis('off')
#     plt.tight_layout()
#     plt.show()

# # Convert image to 256x256 array
# def convert_image_to_array(image_path):
#     try:
#         image = cv2.imread(image_path)
#         if image is not None:
#             image = cv2.resize(image, (256, 256))
#             return img_to_array(image)
#         else:
#             return np.array([])
#     except Exception as e:
#         print(f"[ERROR] Could not process image {image_path}: {e}")
#         return np.array([])

# # Load dataset and return (X, y)
# def load_dataset():
#     image_list = []
#     label_list = []

#     for idx, label_name in enumerate(all_labels):
#         class_dir = os.path.join(dataset_path, label_name)
#         if not os.path.isdir(class_dir):
#             print(f"[WARNING] Skipping missing directory: {class_dir}")
#             continue
#         print(f"[INFO] Loading images from: {label_name}")
#         for img_file in os.listdir(class_dir):
#             img_path = os.path.join(class_dir, img_file)
#             img_array = convert_image_to_array(img_path)
#             if img_array.size > 0:
#                 image_list.append(img_array)
#                 label_list.append(idx)

#     X = np.array(image_list, dtype="float32") / 255.0
#     y = to_categorical(np.array(label_list), num_classes=len(all_labels))
#     return X, y

# # Build the CNN model
# def build_model():
#     model = Sequential([
#         Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=(256, 256, 3)),
#         MaxPooling2D(pool_size=(3, 3)),
#         Conv2D(16, (3, 3), padding="same", activation="relu"),
#         MaxPooling2D(pool_size=(2, 2)),
#         Flatten(),
#         Dense(8, activation="relu"),
#         Dense(3, activation="softmax")  # 3 classes
#     ])
#     model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-4), metrics=['accuracy'])
#     model.summary()
#     return model

# # Main training process
# def main():
#     # Optional: preview_images()

#     X, y = load_dataset()
#     print(f"[INFO] Dataset loaded: {X.shape[0]} samples.")

#     # Train/test/validation split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

#     model = build_model()

#     history = model.fit(
#         X_train, y_train,
#         batch_size=32,
#         epochs=50,
#         validation_data=(X_val, y_val),
#         verbose=1
#     )

#     # Save model
#     model.save(model_save_path)
#     print(f"[INFO] Model saved to: {model_save_path}")

# if __name__ == "__main__":
#     main()


import os
import random
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from matplotlib.image import imread
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Set dataset path
DATASET_PATH = r"C:\Users\lenovo\Documents\Plant-Disease-Detection-main\Dataset"
VISUALIZE_PATH = os.path.join(DATASET_PATH, "Corn___Common_rust")
MODEL_SAVE_PATH = r"C:\Users\lenovo\Documents\Plant-Disease-Detection-main\Model\plant_disease_model.h5"

def visualize_images():
    plt.figure(figsize=(12, 12))
    for i in range(1, 17):
        plt.subplot(4, 4, i)
        plt.tight_layout()
        rand_img = imread(os.path.join(VISUALIZE_PATH, random.choice(os.listdir(VISUALIZE_PATH))))
        plt.imshow(rand_img)
        plt.xlabel(rand_img.shape[1], fontsize=10)
        plt.ylabel(rand_img.shape[0], fontsize=10)
    plt.show()

def convert_image_to_array(image_path):
    try:
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.resize(image, (256, 256))
            return img_to_array(image)
        else:
            return np.array([])
    except Exception as e:
        print(f"Error: {e}")
        return None

def load_dataset():
    image_list = []
    label_list = []
    
    all_labels = ['Corn___Common_rust', 'Grape___Black_rot', 'Rice___Bacterialblight']
    binary_labels = [0, 1, 2]
    root_dir = os.listdir(DATASET_PATH)
    
    for idx, category in enumerate(root_dir):
        category_path = os.path.join(DATASET_PATH, category)
        if not os.path.isdir(category_path):
            continue
        for file in os.listdir(category_path):
            image_path = os.path.join(category_path, file)
            img_arr = convert_image_to_array(image_path)
            if img_arr is not None and img_arr.size > 0:
                image_list.append(img_arr)
                label_list.append(binary_labels[idx])

    return np.array(image_list), np.array(label_list)

def build_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=(256, 256, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Conv2D(16, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(8, activation="relu"))
    model.add(Dense(3, activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])
    model.summary()
    return model

def main():
    print("Visualizing sample images...")
    visualize_images()

    print("Loading and preprocessing dataset...")
    images, labels = load_dataset()

    print(f"Dataset loaded: {images.shape[0]} images.")

    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=10)
    x_train = x_train.astype('float16') / 255.0
    x_test = x_test.astype('float16') / 255.0
    x_train = x_train.reshape(-1, 256, 256, 3)
    x_test = x_test.reshape(-1, 256, 256, 3)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Split validation data
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=10)

    print("Building model...")
    model = build_model()

    print("Training model...")
    history = model.fit(x_train, y_train, batch_size=128, epochs=50, validation_data=(x_val, y_val))

    print(f"Saving model to {MODEL_SAVE_PATH}")
    model.save(MODEL_SAVE_PATH)

if __name__ == "__main__":
    main()
