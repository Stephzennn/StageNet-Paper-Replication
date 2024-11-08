import os
import shutil
import argparse
import random

def move_to_partition(args, patients, partition):
    if not os.path.exists(os.path.join(args.subjects_root_path, partition)):
        os.mkdir(os.path.join(args.subjects_root_path, partition))
    for patient in patients:
        src = os.path.join(args.subjects_root_path, patient)
        dest = os.path.join(args.subjects_root_path, partition, patient)
        shutil.move(src, dest)

def main():
    parser = argparse.ArgumentParser(description='Split data into train and test sets.')
    parser.add_argument('subjects_root_path', type=str, help='Directory containing subject sub-directories.')
    args, _ = parser.parse_known_args()

    folders = os.listdir(args.subjects_root_path)
    folders = list(filter(str.isdigit, folders))

    # Shuffle the folders to ensure random distribution
    random.shuffle(folders)

    # Calculate the number of train and test patients
    num_test_patients = int(len(folders) * 0.99)
    num_train_patients = len(folders) - num_test_patients

    train_patients = folders[:num_train_patients]
    test_patients = folders[num_train_patients:]

    move_to_partition(args, train_patients, "train")
    move_to_partition(args, test_patients, "test")

if __name__ == '__main__':
    main()
