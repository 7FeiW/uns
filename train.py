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

def train_and_predict(model_name="unet",
                      num_epoch=10, batch_size=32, verbose=0, filters=32):
    # folder name to save current run

    if not os.path.exists('models/'):
        os.mkdir('models')

    if not os.path.exists('models/' + model_name):
        os.mkdir('models/' + model_name)

    folder_name = 'models/' + model_name + '/e' + str(num_epoch) + '_b' + str(batch_size) + '_f' + str(filters)
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    # load data
    imgs_train, imgs_mask_train = get_training_npy_data()

    # resize data
    imgs_train = resize_all_images(imgs_train, img_rows, img_cols)
    imgs_mask_train = resize_all_images(imgs_mask_train, img_rows, img_cols)

    imgs_train = imgs_train.astype('float32')

    # pre proccessing
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    # save mean and std
    data = { 'mean': float(mean), 'std' : float(std)}

    with open(folder_name + '/data.json', 'w+') as outfile:
        json.dump(data, outfile)

    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.0  # scale masks to [0, 1]
    imgs_mask_train = np.round(imgs_mask_train)

    imgs_train, imgs_mask_train = unison_shuffled_copies(imgs_train, imgs_mask_train)

    # split data into training set and validation set
    train_set_len = len(imgs_train)
    val_cut_off = int(train_set_len * 0.1)
    test_cut_off = int(train_set_len * 0.3)

    val_set = imgs_train[:val_cut_off]
    val_set_masks = imgs_mask_train[:val_cut_off]

    # set for evlautation 
    test_set = imgs_train[val_cut_off: test_cut_off]
    test_set_masks = imgs_train[val_cut_off: test_cut_off]

    # set fro training
    train_set = imgs_train[test_cut_off:]
    train_set_masks = imgs_mask_train[test_cut_off:]

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
    else:
        print('ooops')

    # set tensorboard
    tensorboard = TensorBoard(log_dir=folder_name + '/logs', histogram_freq=0,  write_graph=True, write_images=False)
    if not os.path.exists(folder_name + '/logs'):
        os.mkdir(folder_name + '/logs')

    # create data generator
    data_gen = ImageDataGenerator(
        rotation_range=20,
        vertical_flip=True,
        horizontal_flip=True)

    val_gen = ImageDataGenerator()

    #model.compile(optimizer=Adadelta(lr=0.1, rho=0.95, epsilon=1e-08), loss=dice_loss2, metrics=[dice_coef2,hard_dice_coef2])
    model.compile(optimizer=Adam(lr=1e-5), loss=dice_loss, metrics=[hard_dice_coef])

    # set model checkpoint
    model_checkpoint = ModelCheckpoint(folder_name + '/models.h5', monitor='val_loss', save_best_only=True)

    # early stop
    EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=0, mode='auto')

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
    evl_score = model.evaluate(val_set, val_set_masks, batch_size=batch_size, verbose=verbose)
    #print("evaluate score:", "loss,metrics")
    print("evaluate score:" , evl_score)

    predicted_val_masks = model.predict(val_set, verbose=verbose)
    predicted_val_masks = np.around(predicted_val_masks)
    shape = predicted_val_masks.shape
    predicted_val_masks_reshaped = np.reshape(predicted_val_masks,(shape[0], shape[1] * shape[2]))

    dice = 0.
    for predicted, val_mask in zip(predicted_val_masks, val_set_masks):
        dice += dice_score(predicted,val_mask)
    
    print('hard dice: ', dice/shape[0])

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

    args = parser.parse_args()
    train_and_predict(model_name=args.model,
                      num_epoch=args.num_epoch,
                      batch_size=args.batch_size,
                      verbose=args.verbose,
                      filters=args.filters)
