from keras import backend as K
from keras.losses import binary_crossentropy, mean_squared_error, kullback_leibler_divergence

def soomth_dice_sorensen_coef_batch(y_true, y_pred):
    '''
    Function to compute dice coef over enitre batch
    '''
    smooth= 1.0
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)
    intersection = K.sum(y_true_flat * y_pred_flat)
    l = K.sum(y_true_flat)
    r = K.sum(y_pred_flat)
    
    if l == 0 and r == 0:
        return 1
    else:
        return (2. * intersection + smooth) / ( l + r + smooth)

def hard_dice_cof_batch(y_true, y_pred):
    '''
    Function to compute dice coef over enitre batch
    '''
    smooth= 1.0
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)
    intersection = K.sum(y_true_flat * y_pred_flat)
    l = K.sum(y_true_flat)
    r = K.sum(y_pred_flat)
    
    if l + r ==  0:
        return 1
    else:
        return (2. * intersection) / (l + r)

def soomth_dice_sorensen_coef(y_true, y_pred):  
    smooth= 1e-5
    #intersection needs to be an array of dimension batch_size
    #the number of overlapping pixels needs to be summed over the axes for channels, rows and cols   
    #intersection = K.sum(y_true * y_pred, axis=(1,2,3))
    intersection = K.sum(y_true * K.greater(y_pred, 0.5), axis=(3,2,1))

    #now we calculate the dice coeff for each image in the batch
    #the returned value is the mean of the dice coefficients calculated for each image

    return K.mean( (2. * intersection + smooth) / (smooth + K.sum(K.greater(y_true,0.5), axis=(3,2,1)) + K.sum(K.greater(y_pred, 0.5), axis=(3,2,1))))

def soomth_dice_sorensen_coef(y_true, y_pred):
    # Workaround for shape bug. For some reason y_true shape was not being set correctly
    y_true.set_shape(y_pred.get_shape())

    # Without K.clip, K.sum() behaves differently when compared to np.count_nonzero()
    y_true_f = K.clip(K.batch_flatten(y_true), K.epsilon(), 1.)
    y_pred_f = K.clip(K.batch_flatten(y_pred), K.epsilon(), 1.)

    intersection = K.sum(y_true_f * y_pred_f, axis=1)
    union = K.sum(y_true_f * y_true_f, axis=1) + K.sum(y_pred_f * y_pred_f, axis=1)
    return K.mean(2 * intersection / union)

def dice_loss(y_true, y_pred):
    return 1.0 - soomth_dice_sorensen_coef_batch(y_true, y_pred)