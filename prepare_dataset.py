# Import dependencies
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import random
import cv2 as cv
import os


REQD_PALMAR_IMGS = 12
REQD_DORSAL_IMGS = 12
PERSONS_REQD = 165


# Set up paths
IMAGES_DIRPATH = "E:/college_project/11K_Hands3/"
METADATA_FILEPATH = "E:/college_project/HandInfo.csv"
DATASET_DIRPATH = "E:/college_project/college_dataset3"
EXTRAS_FOLDER = os.path.join(DATASET_DIRPATH, "extra")
PALMAR_DIRPATH = os.path.join(DATASET_DIRPATH, "palmar")
PALMAR_TRAIN = os.path.join(PALMAR_DIRPATH, "train")
PALMAR_TEST  = os.path.join(PALMAR_DIRPATH, "test")
DORSAL_DIRPATH = os.path.join(DATASET_DIRPATH, "dorsal")
DORSAL_TRAIN = os.path.join(DORSAL_DIRPATH, "train")
DORSAL_TEST  = os.path.join(DORSAL_DIRPATH, "test")



def palmar_train_test_split(n_train):    
    filenames = os.listdir(EXTRAS_FOLDER)

    # Save train images
    train_fnames = random.sample(filenames, n_train)
    for fname in train_fnames:
        os.replace(os.path.join(EXTRAS_FOLDER, fname),
                   os.path.join(os.path.join(PALMAR_TRAIN, fname)))
        
    # Save remaining images as test images
    test_fnames = os.listdir(EXTRAS_FOLDER)
    for fname in test_fnames:
        os.replace(os.path.join(EXTRAS_FOLDER, fname),
                   os.path.join(os.path.join(PALMAR_TEST, fname)))
        


def dorsal_train_test_split(n_train):    
    filenames = os.listdir(EXTRAS_FOLDER)

    # Save train images
    train_fnames = random.sample(filenames, n_train)
    for fname in train_fnames:
        os.replace(os.path.join(EXTRAS_FOLDER, fname),
                   os.path.join(os.path.join(DORSAL_TRAIN, fname)))
        
    # Save remaining images as test images
    test_fnames = os.listdir(EXTRAS_FOLDER)
    for fname in test_fnames:
        os.replace(os.path.join(EXTRAS_FOLDER, fname),
                   os.path.join(os.path.join(DORSAL_TEST, fname)))
        
        
        
def get_augmented_images_and_save_them(label, num_reqd, filename, dst_dirpath):
    # Read contents of the file
    image = cv.imread(os.path.join(IMAGES_DIRPATH, filename))
    image = tf.expand_dims(image, axis=0) # add `batch_size` dimension

    augmented_data_generator = ImageDataGenerator(rotation_range=15,
                                                  zoom_range=[0.9, 1.1])
    aug_iter = augmented_data_generator.flow(image, batch_size=1)
    
    idx = 1
    # Get required augmented images
    while(num_reqd > 0):
        aug_img = aug_iter.next()[0].astype('uint8')
        aug_img_name = f"label-{label}--{idx}_aug.jpg"
        cv.imwrite(os.path.join(dst_dirpath, aug_img_name), aug_img) # save
        
        num_reqd -= 1
        idx += 1
    return idx        



def save_files(label, idx, files, dst_dirpath):
    for fname in files:
        os.replace(os.path.join(IMAGES_DIRPATH, fname),
                   os.path.join(dst_dirpath, f"label-{label}--{idx}.jpg"))
                
        if idx != 12:
            idx += 1
        else:
            break  




# Read dataset metadata
metadata = pd.read_csv(METADATA_FILEPATH)

# No. of individuals (each id denotes an individual)
unique_ids = metadata.id.unique() 

# Loop through each id (person). If number of palmar/dorsal images is less than
# the required number, add augmented palmar/dorsal images in the dataset.
label = 0
for ID in unique_ids[2:]:
    palmar_fnames_list = metadata[(metadata.id == ID) & (metadata.aspectOfHand == 'palmar right')].imageName.values.tolist()
    dorsal_fnames_list = metadata[(metadata.id == ID) & (metadata.aspectOfHand == 'dorsal right')].imageName.values.tolist()
    
    num_palmar_imgs, num_dorsal_imgs = len(palmar_fnames_list), len(dorsal_fnames_list)
    
    if (num_palmar_imgs and num_dorsal_imgs):
        
        # *****************************   FOR PALMAR IMAGES  *************************************** #
                
        if num_palmar_imgs < REQD_PALMAR_IMGS+1:
            # Select a random imagefile and get its augmentations
            rand_fname = random.choice(palmar_fnames_list)
            
            idx = get_augmented_images_and_save_them(label,
                                                     num_reqd = REQD_PALMAR_IMGS - num_palmar_imgs,
                                                     filename=rand_fname, 
                                                     dst_dirpath=EXTRAS_FOLDER)
            
            # Save original imagefiles
            save_files(label=label, idx=idx, files=palmar_fnames_list, dst_dirpath=EXTRAS_FOLDER)
        
        else:
            # Save required number of imagefiles
            save_files(label=label, idx=1, files=palmar_fnames_list, dst_dirpath=EXTRAS_FOLDER)

        palmar_train_test_split(n_train=8)
            
        # ****************************************************************************************** #            
            


        # *****************************   FOR DORSAL IMAGES  *************************************** #
                
        if num_dorsal_imgs < REQD_DORSAL_IMGS+1:
            # Select a random imagefile and get its augmentations
            rand_fname = random.choice(dorsal_fnames_list)
            
            idx = get_augmented_images_and_save_them(label,
                                                     num_reqd = REQD_DORSAL_IMGS - num_dorsal_imgs,
                                                     filename=rand_fname, 
                                                     dst_dirpath=EXTRAS_FOLDER)
            
            # Save original imagefiles
            save_files(label, idx=idx, files=dorsal_fnames_list, dst_dirpath=EXTRAS_FOLDER)
            
        else:
            # Save required original imagefiles
            save_files(label, idx=1, files=dorsal_fnames_list, dst_dirpath=EXTRAS_FOLDER)
        
        dorsal_train_test_split(n_train=8)
        
        # ****************************************************************************************** #
        
        label += 1
        
        # If biometrics of required number of individuals is collected,
        # end the dataset preparation process
        if label == PERSONS_REQD:
            break
