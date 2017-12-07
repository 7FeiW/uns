import os
import numpy as np
from skimage.transform import resize
from skimage.io import imsave, imread
from skimage.util import view_as_blocks
from skimage.exposure import histogram


sample_image_rows = 420
sample_image_cols = 580

img_rows = 64
img_cols = 80

def get_data_set(type):
    print('Loading ' + type + ' Data')
    npy_data_path = 'data/npy'
    train_set_file = os.path.join(npy_data_path, type + '_set.npy')
    train_set_masks = os.path.join(npy_data_path, type + '_set_masks.npy')
    if os.path.exists(train_set_file) and os.path.exists(train_set_masks):
        return np.load(train_set_file), np.load(train_set_masks)

def get_training_npy_data():
    print('Loading Traning Data')
    npy_data_path = 'data/npy'
    samples_npy_file = os.path.join(npy_data_path, 'training_images.npy')
    sample_masks_npy_file = os.path.join(npy_data_path, 'training_image_masks.npy')
    if os.path.exists(samples_npy_file) and os.path.exists(sample_masks_npy_file):
        return np.load(samples_npy_file), np.load(sample_masks_npy_file)

def get_testing_npy_data():
    print('Loading Tesing Data')
    npy_data_path = 'data/npy'
    testing_npy_file = os.path.join(npy_data_path, 'testing_images.npy')
    testing_npy_ids_files = os.path.join(npy_data_path, 'testing_idx.npy')
    if os.path.exists(testing_npy_file) and os.path.exists(testing_npy_ids_files):
        return np.load(testing_npy_file), np.load(testing_npy_ids_files)

def pre_process_training_data(image_path, npy_data_path):
    file_names = os.listdir(image_path)
    sample_names = [file_name for file_name in file_names if 'mask' not in file_name]
    num_samples = len(sample_names)
    
    sample_images = np.ndarray((num_samples, sample_image_rows, sample_image_cols), dtype=np.uint8)
    sample_image_masks = np.ndarray((num_samples, sample_image_rows, sample_image_cols), dtype=np.uint8)

    for idx,sample_name in enumerate(sample_names):
        #if idx%100 == 0:
        #    print('loading', idx, '/',num_samples)
        sample_mask_name = sample_name.split('.')[0] + '_mask.tif'

        #load everything as grey scale image
        sample_image = np.array([imread(os.path.join(image_path, sample_name), as_grey=True)])
        sample_image_mask = np.array([imread(os.path.join(image_path, sample_mask_name), as_grey=True)])

        #add to arrays
        sample_images[idx] = sample_image
        sample_image_masks[idx] = sample_image_mask
    
    npy_data_path = 'data/npy'
    training_npy_file = os.path.join(npy_data_path, 'training_images.npy')
    training_image_masks_npy_file = os.path.join(npy_data_path, 'training_image_masks.npy')

    np.save(training_npy_file, sample_images)
    np.save(training_image_masks_npy_file, sample_image_masks)
    print('Finish Traning Data')


def pre_process_testing_data(image_path, npy_data_path):
    sample_names = os.listdir(image_path)
    num_samples = len(sample_names)
    
    sample_images = np.ndarray((num_samples, sample_image_rows, sample_image_cols), dtype=np.uint8)
    image_ids = np.ndarray((num_samples, ), dtype=np.int32)

    for idx,sample_name in enumerate(sample_names):
        image_ids[idx] = sample_name.split('.')[0]
        #load everything as grey scale image
        sample_image = np.array([imread(os.path.join(image_path, sample_name), as_grey=True)])

        #add to arrays
        sample_images[idx] = sample_image
        
    testing_npy_file = os.path.join(npy_data_path, 'testing_images.npy')
    testing_npy_ids_files = os.path.join(npy_data_path, 'testing_idx.npy')
    np.save(testing_npy_file, sample_images)
    np.save(testing_npy_ids_files, image_ids)


def resize_all_images(images, img_rows, img_cols):
    imgs_p = np.ndarray((images.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(images.shape[0]):
        imgs_p[i] = resize(images[i], (img_rows, img_cols), preserve_range=True,  mode='constant')

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p

def unison_shuffled_copies(list_one, linst_two):
    p = np.random.permutation(len(list_one))
    return list_one[p], linst_two[p]

'''
def find_similar_images(image_filenames, hashfunc = imagehash.dhash):
    images = {}
    for img in sorted(image_filenames):
        hash = hashfunc(imread(img))
        images[hash] = images.get(hash, []) + [img]
    
    for k, img_list in six.iteritems(images):
        if len(img_list) > 1:
            print(" ".join(img_list))

def compute_image_hist(img):
    # Divide the image in blocks and compute per-block histogram
    print(img.shape)
    blocks = view_as_blocks(img, block_shape=(32, 32))
    img_hists = [histogram(block) for block in blocks]
    return np.concatenate(img_hists)

def create_clean_data_sets(image_path):
    file_names = os.listdir(image_path)
    file_names = [file_name for file_name in file_names if 'mask' not in file_name]

    sample_grps = {} 
    for file_name in file_names:
        sample_name =  file_name.split('.')[0]
        pid = int(file_name.split('_')[0])
        if pid not in sample_grps:
            sample_grps[pid] = []
        sample_grps[pid].append(file_name)   
    
    #print(sample_grps)
    for grp in sample_grps:
        for sample in sample_grps[grp]:
            image = imread(os.path.join(image_path, sample), as_grey=True)
            image = resize(image, (256, 256), preserve_range=True,  mode='constant')
            print(compute_image_hist(image))

    #print(sample_grps)
'''

def create_data_sets():
    # split data into training set and validation set
    imgs_train, imgs_mask_train = get_training_npy_data()

    imgs_train, imgs_mask_train = unison_shuffled_copies(imgs_train, imgs_mask_train)

    train_set_len = len(imgs_train)
    val_cut_off = int(train_set_len * 0.1)
    test_cut_off = int(train_set_len * 0.2)

    val_set = imgs_train[:val_cut_off]
    val_set_masks = imgs_mask_train[:val_cut_off]

    # set for evlautation 
    test_set = imgs_train[val_cut_off: test_cut_off]
    test_set_masks = imgs_train[val_cut_off: test_cut_off]

    # set for training
    train_set = imgs_train[test_cut_off:]
    train_set_masks = imgs_mask_train[test_cut_off:]

    npy_data_path = './data/npy/'


    np.save(npy_data_path + 'train_set.npy', train_set)
    np.save(npy_data_path + 'train_set_masks.npy', train_set_masks)


    np.save(npy_data_path + 'dev_set.npy', val_set)
    np.save(npy_data_path + 'dev_set_masks.npy', val_set_masks)

    np.save(npy_data_path + 'test_set.npy', val_set)
    np.save(npy_data_path + 'test_set_masks.npy', val_set_masks)

if __name__ == '__main__':
    create_data_sets()