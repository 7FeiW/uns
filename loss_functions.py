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

    return (2. * intersection + smooth) / (l + r + smooth)

def hard_dice_coef(y_true, y_pred):
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

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)