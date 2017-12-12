import warnings
import json

from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K

from loss_functions import *
from models import *
from submission import *
from math import isnan

from keras.callbacks import TensorBoard

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

def dice_score(im1, im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    if im1.sum() + im2.sum() == 0:
        return 1.0
    else:
        return 2. * intersection.sum() / (im1.sum() + im2.sum())

def iou_score(im1, im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    if im1.sum() + im2.sum() == 0:
        return 1.0
    else:
        return intersection.sum() / (im1.sum() + im2.sum())

def train_and_predict(model_name="unet",
                      num_epoch=10, batch_size=32, verbose=0, filters=32, load_model = 0):
    # folder name to save current run

    if not os.path.exists('models/'):
        os.mkdir('models')

    if not os.path.exists('models/' + model_name):
        os.mkdir('models/' + model_name)

    folder_name = 'models/' + model_name + '/e' + str(num_epoch) + '_b' + str(batch_size) + '_f' + str(filters)
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    # load data
    train_set, train_set_masks = get_data_set('train')
    val_set, val_set_masks = get_data_set('dev')
    test_set, test_set_masks = get_data_set('test')

    # resize data
    train_set = resize_all_images(train_set, img_rows, img_cols)
    train_set_masks = resize_all_images(train_set_masks, img_rows, img_cols)

    val_set = resize_all_images(val_set, img_rows, img_cols)
    val_set_masks = resize_all_images(val_set_masks, img_rows, img_cols)

    test_set = resize_all_images(test_set, img_rows, img_cols)
    test_set_masks = resize_all_images(test_set_masks, img_rows, img_cols)


    # pre proccessing
    train_set = train_set.astype('float32')
    val_set = val_set.astype('float32')
    test_set = test_set.astype('float32')

    mean = np.mean(train_set)  # mean for data centering
    std = np.std(train_set)  # std for data normalization

    # save mean and std
    data = { 'mean': float(mean), 'std' : float(std)}

    with open(folder_name + '/data.json', 'w+') as outfile:
        json.dump(data, outfile)

    # normalizaztion
    train_set -= mean
    train_set /= std

    val_set -= mean
    val_set /= std

    test_set -= mean
    test_set /= std

    # pre process masks
    train_set_masks = train_set_masks.astype('float32')
    train_set_masks /= 255.0  # scale masks to [0, 1]
    train_set_masks = np.round(train_set_masks)

    val_set_masks = val_set_masks.astype('float32')
    val_set_masks /= 255.0  # scale masks to [0, 1]
    val_set_masks = np.round(val_set_masks)

    test_set_masks = test_set_masks.astype('float32')
    test_set_masks /= 255.0  # scale masks to [0, 1]
    test_set_masks = np.round(test_set_masks)

    # get model and compile
    if model_name == 'unet':
        model = unet(img_rows=img_rows, img_cols=img_cols, base_filter_num=filters)
    elif model_name == 'unet_bn':
        model = unet_bn(img_rows=img_rows, img_cols=img_cols, base_filter_num=filters)
    elif model_name == 'unet_res':
        model = unet_res(img_rows=img_rows, img_cols=img_cols, base_filter_num=filters)
    elif model_name == 'unet_res_dp':
        model = unet_res_wide_dp(img_rows=img_rows, img_cols=img_cols, base_filter_num=filters)
    elif model_name == 'fcn':
        model = fcn(img_rows=img_rows, img_cols=img_cols, base_filter_num=filters)
    elif model_name == 'unet_deeper':
        model = unet_deeper(img_rows=img_rows, img_cols=img_cols, base_filter_num=filters)
    elif model_name == 'unet_invert':
        model = unet_invert(img_rows=img_rows, img_cols=img_cols, base_filter_num=filters)
    elif model_name == 'unet_invert_small':
        model = unet_invert_small(img_rows=img_rows, img_cols=img_cols, base_filter_num=filters)
    else:
        print('ooops')

    # set tensorboard
    tensorboard = TensorBoard(log_dir=folder_name + '/logs', histogram_freq=0,  write_graph=True, write_images=False)
    if not os.path.exists(folder_name + '/logs'):
        os.mkdir(folder_name + '/logs')

    # create data generator
    data_gen = ImageDataGenerator(
        rotation_range=30,
        vertical_flip=True,
        horizontal_flip=True)

    val_gen = ImageDataGenerator()

    #model.compile(optimizer=Adadelta(lr=0.1, rho=0.95, epsilon=1e-08), loss=dice_loss2, metrics=[dice_coef2,hard_dice_coef2])
    model.compile(optimizer=Adam(lr=1e-5), loss=dice_loss, metrics=[hard_dice_coef, mean_iou])

    # set model checkpoint
    model_checkpoint = ModelCheckpoint(folder_name + '/models.h5', monitor='val_loss', save_best_only=True)

    # early stop
    EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=0, mode='auto')

    if load_model == 1 and os.path.exists(folder_name + '/models.h5'):
        print("Loaded model from disk")
        model.load_weights(folder_name + '/models.h5')

    if num_epoch > 0:
        # fitting model
        model.fit_generator(data_gen.flow(train_set, train_set_masks, batch_size=batch_size, shuffle=True),
                        samples_per_epoch=len(train_set), epochs=num_epoch, verbose=verbose,
                        callbacks=[model_checkpoint, tensorboard],
                        validation_data=val_gen.flow(val_set, val_set_masks, batch_size=batch_size, shuffle=True),
                        validation_steps=int(len(val_set) / 32))

    #model.fit(train_set, train_set_masks, batch_size=batch_size,
    #          epochs=num_epoch, verbose=verbose, 
    #          shuffle=True,validation_split=0.2,callbacks=[model_checkpoint,tensorboard])

    # evalutation
    evl_score = model.evaluate(test_set, test_set_masks, batch_size=batch_size, verbose=verbose)
    #print("evaluate score:", "loss,metrics")
    print("evaluate score:" , evl_score)

    predicted_test_masks = model.predict(test_set, verbose=verbose)
    predicted_test_masks = np.around(predicted_test_masks)
    shape = predicted_test_masks.shape
    predicted_test_masks_reshaped = np.reshape(predicted_test_masks,(shape[0], shape[1] * shape[2]))

    dice = 0.
    iou = 0.
    for predicted, val_mask in zip(predicted_test_masks, test_set_masks):
        dice += dice_score(predicted,val_mask)
        iou += iou_score(predicted,val_mask)
    print('hard dice: ', dice/shape[0], 'mean of iou:', iou/shape[0])
    print('model summary:')
    print(model.count_params())
    print(model.summary())
    # create testings
    '''
    imgs_test, imgs_id_test = get_testing_npy_data()
    imgs_test = resize_all_images(imgs_test, img_rows, img_cols)
    imgs_test = imgs_test.astype('float32')

    model.load_weights(folder_name + '/models.h5')
    imgs_test -= mean
    imgs_test /= std

    imgs_mask_test = model.predict(imgs_test, verbose=verbose)
    np.save(folder_name + '/imgs_mask_test.npy', imgs_mask_test)

    pred_dir = folder_name + '/preds'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)

    for image, image_id in zip(imgs_mask_test, imgs_id_test):
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imsave(os.path.join(pred_dir, str(image_id) + '_pred.png'), image)

    create_submission(folder_name + '/imgs_mask_test.npy', folder_name + '/submission.csv')
    ''' 

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--is_train', type=str, default='True')
    parser.add_argument('--num_epoch', type=int, default=20, help='')
    parser.add_argument('--batch_size', type=int, default=32, help='')
    parser.add_argument('--model', type=str, default='unet', help='unet,fcn,unet_res,unet_deeper')
    parser.add_argument('--verbose', type=int, default=2, help='0 for desiable, 1 for progress bar, 2 for log')
    parser.add_argument('--filters', type=int, default=32, help='')
    parser.add_argument('--load_model', type=int, default=0, help='')

    args = parser.parse_args()
    train_and_predict(model_name=args.model,
                      num_epoch=args.num_epoch,
                      batch_size=args.batch_size,
                      verbose=args.verbose,
                      filters=args.filters,
                      load_model= args.load_model)
