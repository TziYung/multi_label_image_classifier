from image_loader import *
import os
import argparse
import tensorflow as tf
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", required = True, type = str)
    parser.add_argument("-r", "--ratio", nargs = 3, default = [ 0.8, 0.2, 0.0], type = int)
    parser.add_argument("-b", "--batch_size", default = 4, type = int)
    args = parser.parse_args()
    # Cause this program is using EfficientNet B4, so the size would have to be 380
    img_size = 380  

    # Load data
    data_loader = MultiLabelLoader(args.dir, (img_size, img_size), args.batch_size, tuple(args.ratio))

    data_loader.load_image()

    # Image augmentation
    
    image_augmentation = tf.keras.models.Sequential([
                            # Image rotated between 0.15 * 2pi to -0.15 * 2pi
                            tf.keras.layers.RandomRotation(factor = 0.15),
                            # Image shifted bewtween 0.1% to 0.1% in both height and width
                            tf.keras.layers.RandomTranslation(height_factor = 0.1, width_factor = 0.1),
                            # Image flipped horizontal and vertical
                            tf.keras.layers.RandomFlip(),
                            # Do the equation (x - mean) * factor + mean on each independent channel
                            # The range of factor is 1 - 0.1 to 1 + 0.1
                            tf.keras.layers.RandomContrast(factor = 0.1)
                            ], name = "Augmentation")

    inputs = tf.keras.layers.Input((img_size, img_size, 3))
    augmentation = image_augmentation(inputs)
    efficient_net = tf.keras.applications.EfficientNetB4(include_top = False, weights = "imagenet", input_tensor = augmentation)
    # Freeze the weights
    efficient_net.trainable = False
    # Unfreeze the weights to make last few layers trainable
    for layer in efficient_net.layers[-7:]:
        # Do not train batchnormalization layer
        # The batchnormaliztion get the mean and var from larger data distribution and it serve the purpose of stablizing
        # output of previous layer, shouldn't trian it 
        if not isinstance(tf.keras.layers, tf.keras.layers.BatchNormalization):
            layer.trainable = True
    
    layer = tf.keras.layers.GlobalAveragePooling2D()(efficient_net.output)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.Dropout(0.2)(layer)
    layer = tf.keras.layers.Dense(len(data_loader.class_list))(layer)
    outputs = tf.keras.layers.Activation("sigmoid")(layer)

    model = tf.keras.Model(inputs, outputs)
    
    model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
    model.summary()
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor = "val_loss",
        patience = 5,
        restore_best_weights = True
    )
    model.fit(data_loader.train_data, data_loader.train_label, epochs = 30, validation_data = (data_loader.val_data, data_loader.val_data), callbacks = [early_stop])
    model.save("model.keras")
