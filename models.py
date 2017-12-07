from keras.constraints import maxnorm
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization, Dropout, Add
from keras.models import Model


def conv_block(inputs, filters, kernel_size):
    conv = Conv2D(filters, kernel_size, activation='relu', kernel_constraint=maxnorm(3),
                  kernel_initializer="he_uniform", padding='same')(inputs)
    conv = Dropout(0.1)(conv)
    conv = Conv2D(filters, kernel_size, activation='relu', kernel_constraint=maxnorm(3),
                  kernel_initializer="he_uniform", padding='same')(conv)
    conv = Dropout(0.1)(conv)
    return conv


def conv_bn_block(inputs, filters, kernel_size):
    conv = Conv2D(filters, kernel_size, activation='relu', kernel_constraint=maxnorm(3),
                  kernel_initializer="he_uniform", padding='same')(inputs)
    bn = BatchNormalization()(conv)
    conv = Conv2D(filters, kernel_size, activation='relu', kernel_constraint=maxnorm(3),
                  kernel_initializer="he_uniform", padding='same')(bn)
    bn = BatchNormalization()(conv)
    return bn


def conv_bn_res_block(inputs, filters, kernel_size):
    conv1 = Conv2D(filters, kernel_size, activation='relu', kernel_constraint=maxnorm(3),
                   kernel_initializer="he_uniform", padding='same')(inputs)
    bn1 = BatchNormalization()(conv1)
    conv2 = Conv2D(filters, kernel_size, activation='relu', kernel_constraint=maxnorm(3),
                   kernel_initializer="he_uniform", padding='same')(bn1)
    bn2 = BatchNormalization()(conv2)
    added = Add()([conv1, bn2])
    return added


def conv_bn_res_block_drop(inputs, filters, kernel_size):
    conv1 = Conv2D(filters, kernel_size, activation='relu', kernel_constraint=maxnorm(3),
                   kernel_initializer="he_uniform", padding='same')(inputs)
    bn1 = BatchNormalization()(conv1)
    dp1 = Dropout(0.1)(bn1)
    conv2 = Conv2D(filters, kernel_size, activation='relu', kernel_constraint=maxnorm(3),
                   kernel_initializer="he_uniform", padding='same')(dp1)
    bn2 = BatchNormalization()(conv2)
    added = Add()([conv1, bn2])
    return added


def unet(img_rows=64, img_cols=80, base_filter_num=32):
    '''
    simple unet
    '''
    inputs = Input((img_rows, img_cols, 1))
    conv1 = conv_block(inputs, 32, (3, 3))
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv_block(pool1, 64, (3, 3))
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv_block(pool2, 128, (3, 3))
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv_block(pool3, 256, (3, 3))
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = conv_block(pool4, 512, (3, 3))

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = conv_block(up6, 256, (3, 3))

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = conv_block(up7, 128, (3, 3))

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = conv_block(up8, 64, (3, 3))

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = conv_block(up9, 32, (3, 3))

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    return model

def unet_deeper(img_rows=64, img_cols=80, base_filter_num=32):
    '''
    A deeper unet
    '''
    inputs = Input((img_rows, img_cols, 1))
    conv1 = conv_block(inputs, 32, (3, 3))
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv_block(pool1, 64, (3, 3))
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv_block(pool2, 128, (3, 3))
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv_block(pool3, 256, (3, 3))
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = conv_block(pool4, 512, (3, 3))
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    conv6 = conv_block(pool5, 1024, (3, 3))

    up7 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6), conv5], axis=3)
    conv7 = conv_block(up7, 256, (3, 3))

    up8 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7), conv4], axis=3)
    conv8 = conv_block(up8, 128, (3, 3))

    up9 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8), conv3], axis=3)
    conv9 = conv_block(up9, 64, (3, 3))

    up10 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv9), conv2], axis=3)
    conv10 = conv_block(up10, 32, (3, 3))

    up11 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv10), conv1], axis=3)
    conv11 = conv_block(up11, 32, (3, 3))

    conv12 = Conv2D(1, (1, 1), activation='sigmoid')(conv11)

    model = Model(inputs=[inputs], outputs=[conv12])

    return model

def unet_invert(img_rows=64, img_cols=80, base_filter_num=32):
    '''
    simple unet
    '''
    inputs = Input((img_rows, img_cols, 1))
    conv1 = conv_block(inputs, 256, (3, 3))
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv_block(pool1, 128, (3, 3))
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv_block(pool2, 64, (3, 3))
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv_block(pool3, 32, (3, 3))
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = conv_block(pool4, 16, (3, 3))

    up6 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = conv_block(up6, 32, (3, 3))

    up7 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = conv_block(up7, 64, (3, 3))

    up8 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = conv_block(up8, 128, (3, 3))

    up9 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = conv_block(up9, 256, (3, 3))

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    return model

def unet_invert_small(img_rows=64, img_cols=80, base_filter_num=32):
    '''
    simple unet
    '''
    inputs = Input((img_rows, img_cols, 1))
    conv1 = conv_block(inputs, 128, (3, 3))
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv_block(pool1, 64, (3, 3))
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv_block(pool2, 32, (3, 3))
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv_block(pool3, 16, (3, 3))
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = conv_block(pool4, 8, (3, 3))
    
    up6 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = conv_block(up6, 16, (3, 3))

    up7 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = conv_block(up7, 32, (3, 3))

    up8 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = conv_block(up8, 64, (3, 3))

    up9 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = conv_block(up9, 128, (3, 3))

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    return model


def unet_res(img_rows=64, img_cols=80, base_filter_num=32):
    '''
    unet with res_block
    '''
    inputs = Input((img_rows, img_cols, 1))

    conv1 = conv_bn_res_block(inputs, base_filter_num, (3, 3))
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    base_filter_num *= 2
    conv2 = conv_bn_res_block(pool1, base_filter_num, (3, 3))
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    base_filter_num *= 2
    conv3 = conv_bn_res_block(pool2, base_filter_num, (3, 3))
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    base_filter_num *= 2
    conv4 = conv_bn_res_block(pool3, base_filter_num, (3, 3))
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    base_filter_num *= 2
    conv5 = conv_bn_res_block(pool4, base_filter_num, (3, 3))

    base_filter_num //= 2
    up6 = concatenate([Conv2DTranspose(base_filter_num, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = conv_bn_res_block(up6, 256, (3, 3))

    base_filter_num //= 2
    up7 = concatenate([Conv2DTranspose(base_filter_num, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = conv_bn_res_block(up7, base_filter_num, (3, 3))

    base_filter_num //= 2
    up8 = concatenate([Conv2DTranspose(base_filter_num, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = conv_bn_res_block(up8, base_filter_num, (3, 3))

    base_filter_num //= 2
    up9 = concatenate([Conv2DTranspose(base_filter_num, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = conv_bn_res_block(up9, base_filter_num, (3, 3))

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    return model


def unet_res_wide_dp(img_rows=64, img_cols=80, base_filter_num=32):
    '''
    unet with wide dropout layer
    '''
    inputs = Input((img_rows, img_cols, 1))
    conv1 = conv_bn_res_block_drop(inputs, base_filter_num, (3, 3))
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    base_filter_num *= 2
    conv2 = conv_bn_res_block_drop(pool1, base_filter_num, (3, 3))
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    base_filter_num *= 2
    conv3 = conv_bn_res_block_drop(pool2, base_filter_num, (3, 3))
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    base_filter_num *= 2
    conv4 = conv_bn_res_block_drop(pool3, base_filter_num, (3, 3))
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    base_filter_num *= 2
    conv5 = conv_bn_res_block_drop(pool4, base_filter_num, (3, 3))

    base_filter_num //= 2
    up6 = concatenate([Conv2DTranspose(base_filter_num, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = conv_bn_res_block_drop(up6, base_filter_num, (3, 3))

    base_filter_num //= 2
    up7 = concatenate([Conv2DTranspose(base_filter_num, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = conv_bn_res_block_drop(up7, base_filter_num, (3, 3))

    base_filter_num //= 2
    up8 = concatenate([Conv2DTranspose(base_filter_num, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = conv_bn_res_block_drop(up8, base_filter_num, (3, 3))

    base_filter_num //= 2
    up9 = concatenate([Conv2DTranspose(base_filter_num, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = conv_bn_res_block_drop(up9, base_filter_num, (3, 3))

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    return model


def unet_bn(img_rows=64, img_cols=80, base_filter_num=32):
    '''
    unet with res_block
    '''
    inputs = Input((img_rows, img_cols, 1))
    conv1 = conv_bn_block(inputs, 32, (3, 3))
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv_bn_block(pool1, 64, (3, 3))
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv_bn_block(pool2, 128, (3, 3))
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv_bn_block(pool3, 256, (3, 3))
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = conv_bn_block(pool4, 512, (3, 3))
    #conv5 = Dropout(0.5)(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = conv_bn_block(up6, 256, (3, 3))

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = conv_bn_block(up7, 128, (3, 3))

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = conv_bn_block(up8, 64, (3, 3))

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = conv_bn_block(up9, 32, (3, 3))

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    return model


def fcn(img_rows=64, img_cols=80, base_filter_num=32):
    '''
    simple fcn, whcih means Encoder_Decoder style net
    '''
    inputs = Input((img_rows, img_cols, 1))
    conv1 = conv_block(inputs, 32, (3, 3))
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv_block(pool1, 64, (3, 3))
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv_block(pool2, 128, (3, 3))
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv_block(pool3, 256, (3, 3))
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = conv_block(pool4, 512, (3, 3))

    up6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5)
    conv6 = conv_bn_block(up6, 256, (3, 3))

    up7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6)
    conv7 = conv_bn_block(up7, 128, (3, 3))

    up8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7)
    conv8 = conv_bn_block(up8, 64, (3, 3))

    up9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8)
    conv9 = conv_bn_block(up9, 32, (3, 3))

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    return model