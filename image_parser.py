import glob
import numpy as np
import cv2
import os

# Parameters to set
number_per_char = 6 # How many times did you draw the same char in a sequence
chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' # What chars have you drawn
nr_of_train = 4 # How many chars do you want to use from number_per_char to train your model
lines_to_use = 26 # How many lines are filled on all of your images (templates)
out_dir = "tengwar" # Name a dir where you want to save your train and test data

# Check input
if nr_of_train > number_per_char:
    print("You want to train your model with more data then you drew your self. Nice try bro!")
    exit(1)

if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

# Create the sequence of chars how the images will come after each
char_seq = np.squeeze([number_per_char*[ch] for ch in chars]).flatten()
print(char_seq)
img_dir = "./imgs/"
out_test_dir = './tengwar/test/'
out_train_dir = './tengwar/train/'

if not os.path.isdir(out_test_dir):
    os.mkdir(out_test_dir)
if not os.path.isdir(out_train_dir):
    os.mkdir(out_train_dir)

images = [cv2.imread(path,0) for path in glob.glob(img_dir+"*.jpg")]
imgcntr = 0
char_test_cntr = 0
char_train_cntr = 0

for img in images:
    height, width = np.array(img).shape
    cell_height = int(height / 9 + 0.5)
    cell_width = int(width / 6 + 0.5)

    size_zero = 15
    filt = np.ones([cell_height, cell_width])
    filt[0:size_zero, 0:] = 0
    filt[-size_zero:, 0:] = 0
    filt[0:, 0:size_zero] = 0
    filt[0:, -size_zero:] = 0

    cur_img = np.uint8(np.array(img))
    cur_img = 255-cur_img
    for idx in range(9):
        if imgcntr*9+idx >= lines_to_use:
            break

        for idy in range(6):
            char = cur_img[cell_height*idx:cell_height*(idx+1),cell_width*idy:cell_width*(idy+1)]
            char = np.uint8(char*filt[0:char.shape[0],0:char.shape[1]])
            char = cv2.resize(char, (64, 64), interpolation=cv2.INTER_CUBIC)
            outfile = ''

            print(imgcntr*9*6+idx*6+idy)
            cur_train_dir = out_train_dir + char_seq[int(imgcntr*9*6+idx*6+idy)] + '/'
            print(cur_train_dir)
            cur_test_dir = out_test_dir + char_seq[imgcntr*9*6+idx*6+idy] + '/'
            print(cur_test_dir)

            if not os.path.isdir(cur_train_dir):
                os.mkdir(cur_train_dir)

            if not os.path.isdir(cur_test_dir):
                os.mkdir(cur_test_dir)

            if (imgcntr*9*6+idx*6+idy)%number_per_char >= nr_of_train:
                outfile = cur_test_dir+str(char_test_cntr+1)+".png"
                char_test_cntr +=1
                char_train_cntr = 0
            else:
                outfile = cur_train_dir + str(char_train_cntr + 1) + ".png"
                char_train_cntr += 1
                char_test_cntr = 0
            print(outfile)
            cv2.imwrite(outfile,char)
    imgcntr += 1