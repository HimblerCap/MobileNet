import tensorflow as tf


#### Definición convolucion
def convolucion_2D(input, filters, kernel):
    x = tf.keras.layers.Conv2D(filters, kernel, padding='same', name='conv1')(input)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.ReLU()(x)


#### Definicion Depthwise
def depth_wise_conv(input,stride,filter):
    x = tf.keras.layers.DepthwiseConv2D((3,3), strides=stride)(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filter, (1,1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.ReLU()(x)


#### Arquitectura general}
def mobileNet(input, clases):
    # Arquitectura
    x = convolucion_2D(input, 32, (3,3))
    x = depth_wise_conv(x, 64, (1,1))
    x = depth_wise_conv(x, 128, (2,2))
    x = depth_wise_conv(x, 128, (1,1))
    x = depth_wise_conv(x, 256, (2,2))
    x = depth_wise_conv(x, 256, (1,1))
    for i in range(5):
        x = depth_wise_conv(x, 512, (1,1))
    x = depth_wise_conv(x, 1024, (2,2))
    x = depth_wise_conv(x, 1024, (2,2))
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    output = tf.keras.layers.Dense(clases, activation='softmax')(x)
    return tf.keras.Model(input, output)


def train_model():
    """ Entrenamiento del modelo
    """
    pass


def prediccion():
    """ Realizar las predicciones
    """
    pass

