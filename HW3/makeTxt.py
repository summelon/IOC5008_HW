import os
import random
import h5py
from PIL import Image
from tqdm import tqdm


# Convert bbox to defined format
# Bbox format is ['label', 'left', 'top', 'width', 'height']
# Img_size format is (width, height)
# Return tuple of (x_center, y_center, width, height)
def convert(img_size, bbox):
    width = bbox[3] / img_size[0]
    height = bbox[4] / img_size[1]
    x_center = (bbox[1] + bbox[3] / 2.0) / img_size[0]
    y_center = (bbox[2] + bbox[4] / 2.0) / img_size[1]

    return (x_center, y_center, width, height)


# Get the index from listdir()[:-4], use the index to fetch the bboxes
# For output list[num_of_labels], use rows[num] to fetch bbox of each labels
# Bboxes in rows format is [['label', 'left', 'top', 'width', 'height'], [...], ...]
def get_bbox_rows(index, database):
    database_index = int(index) - 1
    item = database['digitStruct/bbox'][database_index].item()
    num_labels = len(database[item]['label'])
    rows = [[] for _ in range(num_labels)]
    for key in ['label', 'left', 'top', 'width', 'height']:
        attr = database[item][key]
        _ = [rows[i].append(database[attr.value[i].item()].value[0][0])
             for i in range(num_labels)
             ] if num_labels > 1 else [rows[0].append(attr.value[0][0])]
    return rows

# Write labels from .h5 file into .txt file


def write_label_txt(index, hdf5_file_path, txt_save_path, img_folder_path):
    hdf5_file = h5py.File(hdf5_file_path, 'r')
    rows_for_write = get_bbox_rows(index, hdf5_file)
    num_write_rows = len(rows_for_write)
    if num_write_rows != 0:
        label_file_txt = open(txt_save_path+index+'.txt', 'w')
        img_path = img_folder_path + index + '.png'
        img_size = Image.open(img_path).size
        for row_number in range(num_write_rows):
            row = rows_for_write[row_number]
            converted_data = convert(img_size, row)
            if row[0] == 10:
                row[0] = 0
            string_for_write = "{:.0f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                row[0], converted_data[0], converted_data[1],
                converted_data[2], converted_data[3])
            label_file_txt.write(string_for_write)
        label_file_txt.close()

    return True

# --------------------------------------------------------------------------


# Train & validation ratio
val_percent = 0.1

train_img_path = './data/images/train/'
val_img_path = './data/images/val/'
save_img_path_txt_path = './data/'
train_label_path = './data/labels/train/'
val_label_path = './data/labels/val/'
h5_file_path = './train/digitStruct.mat'


total_training_img_name = os.listdir(train_img_path)

num = len(total_training_img_name)
img_list = range(num)
train_dataset = random.sample(img_list, int(num*(1-val_percent)))
val_dataset = random.sample(img_list, int(num*val_percent))


train_txt = open(save_img_path_txt_path+'SVHN_train.txt', 'w')
val_txt = open(save_img_path_txt_path+'SVHN_val.txt', 'w')


with tqdm(total=len(img_list)) as pbar:
    for i in img_list:
        name = total_training_img_name[i] + '\n'
        if i in train_dataset:
            train_txt.write(train_img_path+name)
            write_label_txt(total_training_img_name[i][:-4], h5_file_path,
                            train_label_path, train_img_path)
        else:
            val_txt.write(val_img_path+name)
            write_label_txt(total_training_img_name[i][:-4], h5_file_path,
                            val_label_path, val_img_path)

        pbar.update()


print('Done.')

train_txt.close()
val_txt.close()
