# Import dependencies
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import random
import cv2 as cv
import os


# Set up paths
dataset_dirpath = "E:/college_project/dataset/"
metadata_fpath = "E:/college_project/HandInfo.csv"
new_dataset_dirpath = "E:/college_project/new_dataset/"


REQD_PALMAR_IMGS = 12
REQD_DORSAL_IMGS = 16
PERSONS_REQD = 165
 

def save_files(ID, idx, files, hand_type):
    for fname in files:
        os.rename(src=os.path.join(dataset_dirpath, fname),
                  dst=os.path.join(new_dataset_dirpath, f"{hand_type}/", f"{ID}_{idx}.jpg"))
                
        if idx != 12:
            idx += 1
        else:
            break        
            

def get_augmented_images_and_save_them(ID, num_reqd, image_filename, hand_type):
    # Read the contents of the imagefile
    image = cv.imread(os.path.join(dataset_dirpath, image_filename))
    image = tf.expand_dims(image, axis=0) # add `batch_size` dimension

    # Instantiate augmented image generator
    augmented_data_generator = ImageDataGenerator(rotation_range=15,
                                                  zoom_range=[0.9, 1.1])
    
    aug_iter = augmented_data_generator.flow(image, batch_size=1)
    
    idx = 1
    # Get required no. augmented images
    while(num_reqd > 0):
        img = aug_iter.next()[0].astype('uint8') # augmented image
        # Assign a name to the augmented image
        fname = os.path.join(new_dataset_dirpath, f"{hand_type}/", f"{ID}_{idx}_aug.jpg")
        # Save the augmented image
        cv.imwrite(fname, img)
        
        num_reqd -= 1
        idx += 1
    return idx




# Read file containing metadata of the dataset images
metadata = pd.read_csv(metadata_fpath)


# Each id denotes an individual. Images with same id belongs to the same person.
unique_ids = metadata.id.unique() 


# Loop through each id (person). If number of palmar/dorsal images is less than
# the required number (12 and 16 respectively), add augmented palmar/dorsal
# images in the dataset. For this, select a random image and get its augmentations.
person_num = 0
for ID in unique_ids[112: 117]:
    # Get inner-hand image names list
    palmar_img_fnames = metadata[(metadata.id == ID) & (metadata.aspectOfHand == 'palmar right')].imageName.values.tolist()
    # Ger upper-hand image names list
    dorsal_img_fnames = metadata[(metadata.id == ID) & (metadata.aspectOfHand == 'dorsal right')].imageName.values.tolist()
    
    # Number of palmar and dorsal images of the present id
    num_palmar_imgs, num_dorsal_imgs = len(palmar_img_fnames), len(dorsal_img_fnames)
    
    if (num_palmar_imgs and num_dorsal_imgs):
            
        # *****************************   FOR PALMAR IMAGES  *************************************** #
        
        if num_palmar_imgs < REQD_PALMAR_IMGS+1:
            # Get a random palmar imagefile from which we will be getting augmented images
            rand_img_fname = random.choice(palmar_img_fnames)
            
            # Get augmentations of the selected random palmar imagefile. Assign
            # each augmented image a new name and save them.
            idx = get_augmented_images_and_save_them(ID,
                                                     num_reqd = REQD_PALMAR_IMGS - num_palmar_imgs,
                                                     image_filename=rand_img_fname, 
                                                     hand_type="palmar")
            
            # Save original palmar imagefiles after assigning each file a new name
            save_files(ID=ID, idx=idx, files=palmar_img_fnames, hand_type="palmar")
        
        else:
            # Save first 12 of the original palmar imagefiles after assigning
            # each file a new name
            save_files(ID=ID, idx=1, files=palmar_img_fnames, hand_type="palmar")
            
        # ****************************************************************************************** #            
            


        # *****************************   FOR DORSAL IMAGES  *************************************** #
        if num_dorsal_imgs < REQD_DORSAL_IMGS+1:
            # Get a random dorsal imagefile from which we well be getting augmented images
            rand_img_fname = random.choice(dorsal_img_fnames)
            
            # Get augmentations of the selected random dorsal imagefile. Assign
            # each augmented image a new name and save them.
            idx = get_augmented_images_and_save_them(ID,
                                                     num_reqd = REQD_DORSAL_IMGS - num_dorsal_imgs,
                                                     image_filename=rand_img_fname, 
                                                     hand_type="dorsal")
            
            # Save original dorsal imagefiles after assigning each file a new name
            save_files(ID=ID, idx=idx, files=dorsal_img_fnames, hand_type="dorsal")
            
        else:
            # Save first 12 of the original dorsal imagefiles after assigning
            # each file a new name
            save_files(ID=ID, idx=1, files=dorsal_img_fnames, hand_type="dorsal")
        
        # ****************************************************************************************** #
        
        person_num += 1
        
        # If biometrics of required number of individuals is collected,
        # exit the loop
        if person_num == PERSONS_REQD:
            break
