from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os
def process(path):
        imagePaths = list(paths.list_images(path))
        #print("image path=",imagePaths)
        data = []
        labels = []
        # loop over the image paths
        for imagePath in imagePaths:
                # extract the class label from the filename
                label = imagePath.split(os.path.sep)[-2]
                #print("Label for images",label)
                # load the input image (224x224) and preprocess it
                image = load_img(imagePath, target_size=(224, 224))
                image = img_to_array(image)
                image = preprocess_input(image)
                # update the data and labels lists, respectively
                data.append(image)
                labels.append(label)
                #print("Labels:",labels)
        # convert the data and labels to NumPy arrays
        data = np.array(data, dtype="float32")
        labels = np.array(labels)
        #print("Data===",data)
        #print("Labels==",labels)
        #The next step is to load the pre-trained model and customize it according to our problem. So we just remove the top layers of this pre-trained model and add few layers of our own. As you can see the last layer has two nodes as we have only two outputs. This is called transfer learning.

        baseModel = MobileNetV2(weights="imagenet", include_top=False,
                input_shape=(224, 224, 3))
        # construct the head of the model that will be placed on top of the
        # the base model
        headModel = baseModel.output
        headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(128, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)
        headModel = Dense(5, activation="softmax")(headModel)

        # place the head FC model on top of the base model (this will become
        # the actual model we will train)
        model = Model(inputs=baseModel.input, outputs=headModel)
        # loop over all layers in the base model and freeze them so they will
        # *not* be updated during the first training process
        for layer in baseModel.layers:
                layer.trainable = False
        lb = LabelBinarizer()
        labels = lb.fit_transform(labels)
        #labels = to_categorical(labels)
        # partition the data into training and testing splits using 80% of
        # the data for training and the remaining 20% for testing
        #nsamples, nx, ny = data.shape
        #d2_train_dataset = train_dataset.reshape((nsamples,nx*ny))
        (trainX, testX, trainY, testY) = train_test_split(data, labels,test_size=0.33, stratify=labels, random_state=42)
        # construct the training image generator for data augmentation
        aug = ImageDataGenerator(
                rotation_range=20,
                zoom_range=0.15,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.15,
                horizontal_flip=True,
                fill_mode="nearest")
        INIT_LR = 1e-4
        EPOCHS = 20
        BS = 32
        print("[INFO] compiling model...")
        opt = Adam(lr=INIT_LR)
        model.compile(loss="binary_crossentropy", optimizer=opt,
                metrics=["accuracy"])
        # train the head of the network
        print("[INFO] training head...")
        H = model.fit(
                aug.flow(trainX, trainY, batch_size=BS),
                steps_per_epoch=len(trainX) // BS,
                validation_data=(testX, testY),
                validation_steps=len(testX) // BS,
                epochs=EPOCHS)
        N = EPOCHS
        print(H.history.keys())
        plt.plot(H.history['acc'])
        plt.plot(H.history['val_acc'])
        #plt.title('model accuracy')
        plt.title('Training and validation accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'Validation'], loc='upper left')
        plt.savefig('results/Training and validation accuracy.png') 
        plt.pause(5)
        plt.show(block=False)
        plt.close()
        # summarize history for loss
        plt.plot(H.history['loss'])
        plt.plot(H.history['val_loss'])
        #plt.title('model loss')
        plt.title('Training and validation Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'Validation'], loc='upper left')
        plt.savefig('results/Training and validation Loss.png') 
        plt.pause(5)
        plt.show(block=False)
        plt.close()
        model.save('soil_typemodel.h5')
process("./Dataset")

