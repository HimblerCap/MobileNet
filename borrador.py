import tensorflow as tf


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense()
])


# 
x = tf.keras.layers.DepthwiseConv2D((3,3), strides=stride)(input)
x = tf.keras.layers.BatchNormalization()(x)

# Canal 1 
y = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.Conv2D(filter, (1,1), padding='same')(x)
x = tf.keras.layers.BatchNormalization()(x)

tf.keras.Model(inputs=[], )
