import shutil
import random
import os
# import argparse
import warnings
from rich.progress import track

def change_labels(label):
    if label == 0 or label == 2:
        return 'discarded'
    elif label == 3:
        return 'discarded' # return 0 if need to use center_R
    elif label == 1:
        return 1
    else:
        return label

def process_file(input_file_path, output_file_path):
    with open(input_file_path, 'r') as file:
        lines = file.readlines()
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if parts:
            label = int(parts[0])
            new_label = change_labels(label)
            if new_label == 0:  # For bboxes only
                bbox = parts[1:5]  # x_center, y_center, width, height
                # Add 8 groups of "0 0 0" for the keypoints
                new_line = f"{new_label} {' '.join(bbox)} {'0 0 0 ' * 8}".rstrip()
                new_lines.append(new_line)
            elif new_label != 'discarded':
                bbox = parts[1:5]  # x_center, y_center, width, height
                
                # Handle keypoints properly
                if len(parts) > 5:
                    keypoints = []
                    kp_data = parts[5:]
                    
                    # Extract first 8 keypoints in proper format
                    i = 0
                    count = 0
                    while i < len(kp_data) and count < 8:
                        if i+2 < len(kp_data):
                            # Format: x y visibility
                            x = kp_data[i]
                            y = kp_data[i+1]
                            vis = kp_data[i+2]
                            keypoints.extend([x, y, vis])
                            count += 1
                        i += 3
                        
                    new_line = f"{new_label} {' '.join(bbox)} {' '.join(keypoints[:24])}"
                    new_lines.append(new_line)
                else:
                    # Skip lines without keypoints
                    continue
            else:
                continue
    
    with open(output_file_path, 'w') as file:
        for line in new_lines:
            file.write('\n'.join(new_lines))

def check_for_label_1(input_file_path):
    """
    Check if the label file contains any entries with label value 1.
    Returns True if it does, False otherwise.
    """
    with open(input_file_path, 'r') as file:
        lines = file.readlines()
    
    for line in lines:
        parts = line.strip().split()
        if parts and int(parts[0]) == 1:
            return True
    
    return False

def split_label(dataset_all_path, dataset_split_path, train_percent=0.9, val_percent=0.1, test_mode=0):
    image_original_path = dataset_all_path + '/images/'
    label_original_path = dataset_all_path + '/labels/'
    train_image_path = dataset_split_path + '/images/train/'
    train_label_path = dataset_split_path + '/labels/train/'
    txt_train_file = dataset_split_path + '/train.txt'
    val_image_path = dataset_split_path + '/images/val/'
    val_label_path = dataset_split_path + '/labels/val/'
    txt_val_file = dataset_split_path + '/val.txt'

    if test_mode:
        warnings.warn("You choose creat test dataset, if you have a small dataset, we do not recommend you to do this!")
        test_image_path = dataset_split_path + '/images/test/'
        test_label_path = dataset_split_path + '/labels/test/'
        txt_test_file = dataset_split_path + '/test.txt'

    if not os.path.exists(dataset_split_path):
        warnings.warn("Could no find saving_path,Creat one!")
    if not os.path.exists(train_image_path):
        os.makedirs(train_image_path)
    if not os.path.exists(train_label_path):
        os.makedirs(train_label_path)
        file = open(txt_train_file, 'w')
        file.write("")
        file.close()

    if not os.path.exists(val_image_path):
        os.makedirs(val_image_path)

    if not os.path.exists(val_label_path):
        os.makedirs(val_label_path)
        file = open(txt_val_file, 'w')
        file.write("")
        file.close()

    if test_mode:
        if not os.path.exists(test_image_path):
            os.makedirs(test_image_path)
        if not os.path.exists(test_label_path):
            os.makedirs(test_label_path)
            file = open(txt_test_file, 'w')
            file.write("")
            file.close()

    total_txt = os.listdir(label_original_path)
    num_txt = len(total_txt)
    list_all_txt = range(num_txt)
    num_train = int(num_txt * train_percent)
    num_val = int(num_txt * val_percent)
    train = random.sample(list_all_txt, num_train)
    val_test = [i for i in track(list_all_txt) if not i in train]
    if test_mode:
        val = random.sample(val_test, num_val)
    else:
        val = val_test
    file_train = open(txt_train_file, 'w')
    file_val = open(txt_val_file, 'w')
    if test_mode:
        file_test = open(txt_test_file, 'w')
    if test_mode:
        print("train:{}, val:{}, test:{}".format(len(train), len(val), len(val_test) - len(val)))
    else:
        print("train:{}, val:{}".format(len(train), len(val)))
        
    images_skipped = 0
    
    for i in track(list_all_txt):
        name = total_txt[i][:-4]
        srcImage = image_original_path + name + '.jpg'
        srcLabel = label_original_path + name + '.txt'
        if not os.path.exists(srcImage):
            srcImage = image_original_path + name + '.png'
        if not os.path.exists(srcImage):
            print("Could not find image: ", srcImage)
            continue
        if not os.path.exists(srcLabel):
            print("Could not find label: ", srcLabel)
            continue
            
        # Check if the label file contains label 1
        if not check_for_label_1(srcLabel):
            images_skipped += 1
            continue
            
        if i in train:
            dst_train_Image = train_image_path + name + os.path.splitext(srcImage)[1]
            dst_train_Label = train_label_path + name + '.txt'
            shutil.copyfile(srcImage, dst_train_Image)
            process_file(srcLabel, dst_train_Label)
            file_train.write(str(name + os.path.splitext(srcImage)[1] + '\n'))
        elif i in val:
            dst_val_Image = val_image_path + name + os.path.splitext(srcImage)[1]
            dst_val_Label = val_label_path + name + '.txt'
            shutil.copyfile(srcImage, dst_val_Image)
            process_file(srcLabel, dst_val_Label)
            file_val.write(str(name + os.path.splitext(srcImage)[1] + '\n'))
        else:
            dst_test_Image = test_image_path + name + os.path.splitext(srcImage)[1]
            dst_test_Label = test_label_path + name + '.txt'
            shutil.copyfile(srcImage, dst_test_Image)
            process_file(srcLabel, dst_test_Label)
            file_test.write(str(name + os.path.splitext(srcImage)[1] + '\n'))

    file_train.close()
    file_val.close()
    if test_mode:
        file_test.close()
    
    print(f"Total images skipped (no label 1): {images_skipped}")
        
if __name__ == "__main__":
    dataset_all_path = "render"
    dataset_split_path = "render_split"
    split_label(dataset_all_path, dataset_split_path)