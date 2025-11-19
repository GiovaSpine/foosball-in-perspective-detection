import glob
from collections import defaultdict
from config import *
from utility import *


class SetsDivision:
    def __init__(self, train_len: int, val_len: int, test_len: int):
        '''
        '''
        self.train = []
        self.val = []
        self.test = []
        self.train_len = train_len
        self.val_len = val_len
        self.test_len = test_len
        self.index = 0
    
    def add(self, image_path: str):
        '''
        '''
        for _ in range(3):
            if self.index == 0 and len(self.train) < self.train_len:
                self.train.append(image_path)
                self.index = 1
                return
            elif self.index == 1 and len(self.val) < self.val_len:
                self.val.append(image_path)
                self.index = 2
                return
            elif self.index == 2 and len(self.test) < self.test_len:
                self.test.append(image_path)
                self.index = 0
                return
            else:
                self.index = (self.index + 1) % 3

    def full(self):
        '''
        '''
        result = (len(self.train) == self.train_len) and(len(self.val) == self.val_len) and (len(self.test) == self.test_len)
        return result
    
    def save(self):
        '''
        '''
        def save_paths_to_txt(paths_list, output_file):
            '''
            '''
            with open(output_file, 'w') as f:
                for path in paths_list:
                    f.write(f"{path}\n")
            print(f"Saved image paths in {output_file}")
        
        save_paths_to_txt(self.train, TRAIN_TXT_DIRECTORY)
        save_paths_to_txt(self.val, VALIDATION_TXT_DIRECTORY)
        save_paths_to_txt(self.test, TEST_TXT_DIRECTORY)



def divide_training_set(k: int, train_len: int, val_len: int, test_len: int) -> None:
    '''
    Given a k (number of clusters) greater than 3, and a split for the default + added
    dataset, as train_len, val_len, test_len, it produces train.txt, val.txt, test.txt,
    by taking 3 images at a time from each cluster, in the hope to have different images
    in each set.
    The train set will have all the augmented images.
    The files train.txt, val.txt, test.txt will contain paths of the images, and will be
    used by yolo.

    Parameters:
    k (int): The clustering analyisis/division done for that k (number of clusters)
    train_len (int): The split of the default + added dataset for the train set
    val_len (int): The split of the default + added dataset for the validation set
    test_len (int): The split of the default + added dataset for the test set

    Returns:
    None
    '''
    if k < 3 or k > MAX_N_CLUSTERS:
        raise ValueError(f"Warning: k must be an integer between 3 and {MAX_N_CLUSTERS}")

    # of course the validation and test set can't have augmented images
    # so we will split the default + added dataset for the 3 sets
    # and the train set will have all the augmented images
    all_clustering_labels = load_all_clustering_label(ADDED_CLUSTERING_DIRECTORY)

    # dictionary to group images (cluster_id -> [list of image names])
    clusters = defaultdict(list)

    sets_division = SetsDivision(train_len, val_len, test_len)

    for img_name, values in all_clustering_labels.items():
        # we have to group based on the value of the element of index (k - MIN_N_CLUSTERS)
        # of the labels list
        index = k - MIN_N_CLUSTERS
        key = values[index]
        clusters[key].append(img_name)
    
    condition = True
    while(condition):
        for cluster_id in range(0, k):
            if len(clusters[cluster_id]) != 0:
                # remove an image only if the cluster have at least one
                image_name = clusters[cluster_id].pop()  
                image_path = find_image_path(image_name, ADDED_IMAGES_DATA_DIRECTORY)
                sets_division.add(image_path)

        condition = not sets_division.full()

    print("Divided the default + added dataset into:")
    print("- Training set:", len(sets_division.train))
    print("- Validation set:", len(sets_division.val))
    print("- Test set:", len(sets_division.test), "\n")
    sets_division.save()

    print("\nAdding augmented images into the train set...")
    aug_image_form = get_augmented_image_name("").split("_")

    images = glob.glob(os.path.join(AUGMENTED_IMAGES_DATA_DIRECTORY, f"{aug_image_form[0]}_*_*{aug_image_form[2]}"))
    for file in images:
        with open(TRAIN_TXT_DIRECTORY, 'a' ) as f:
            f.write(f"{file}\n")

    print(f"Saved augmented image paths in {TRAIN_TXT_DIRECTORY}")



divide_training_set(k=6, train_len=499, val_len=299, test_len=200)