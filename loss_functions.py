from keras import backend as K

def dice_coef(y_true, y_pred):
    '''
    Function to compute dice coef over enitre batch
    '''
    smooth = 1.0
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)
    intersection = K.sum(y_true_flat * y_pred_flat)
    l = K.sum(y_true_flat)
    r = K.sum(y_pred_flat)

    if l == 0 and r == 0:
        return 1
    else:
        return (2. * intersection + smooth) / (l + r + smooth)


def hard_dice_cof(y_true, y_pred):
    '''
    Function to compute dice coef over enitre batch
    '''
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)
    intersection = K.sum(y_true_flat * y_pred_flat)
    l = K.sum(y_true_flat)
    r = K.sum(y_pred_flat)

    if l + r == 0:
        return 1
    else:
        return (2. * intersection) / (l + r)


def dice_coef2(y_true, y_pred):
    smooth = 1.0

    # Workaround for shape bug. For some reason y_true shape was not being set correctly
    y_true.set_shape(y_pred.get_shape())

    # Without K.clip, K.sum() behaves differently when compared to np.count_nonzero()
    y_true_f = K.clip(K.batch_flatten(y_true), K.epsilon(), 1.)
    y_pred_f = K.clip(K.batch_flatten(y_pred), K.epsilon(), 1.)

    intersection = K.sum(y_true_f * y_pred_f, axis=1)
    union = K.sum(y_true_f * y_true_f, axis=1) + K.sum(y_pred_f * y_pred_f, axis=1)

    return K.mean(2 * intersection / union)

def hard_dice_coef_2(y_true, y_pred):
    # Workaround for shape bug. For some reason y_true shape was not being set correctly
    y_true.set_shape(y_pred.get_shape())

    # Without K.clip, K.sum() behaves differently when compared to np.count_nonzero()
    y_true_f = K.batch_flatten(y_true)
    y_pred_f = K.batch_flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f, axis=1)
    union = K.sum(y_true_f * y_true_f, axis=1) + K.sum(y_pred_f * y_pred_f, axis=1)
    return K.mean(2 * intersection / union)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def dice_loss2(y_true, y_pred):
    return 1.0 - dice_coef2(y_true, y_pred)