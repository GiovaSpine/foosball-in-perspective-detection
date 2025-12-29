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
        
    def add(self, cluster: list):
        '''
        '''

        paths = []
        for image_name in cluster:
            image_path = find_image_path(image_name, IMAGES_DATA_DIRECTORY, ADDED_IMAGES_DATA_DIRECTORY)
            if image_path == None:
                raise FileNotFoundError(f"Error: image {image_name} not found")
            paths.append(image_path)

        for _ in range(3):
            if self.index == 0 and len(self.train) < self.train_len:
                # we certaintly change self.train, but how ?
                if len(paths) > self.train_len - len(self.train):
                    # we have to split paths, because it will fill self.train
                    paths_aux = paths[0:self.train_len - len(self.train)]
                    paths = paths[self.train_len - len(self.train):]
                    self.train += paths_aux
                    self.index = 1
                else:
                    self.train += paths
                    self.index = 1
                    return
            elif self.index == 1 and len(self.val) < self.val_len:
                # we certaintly change self.val, but how ?
                if len(paths) > self.val_len - len(self.val):
                    # we have to split paths, because it will fill self.val
                    paths_aux = paths[0:self.val_len - len(self.val)]
                    paths = paths[self.val_len - len(self.val):]
                    self.val += paths_aux
                    self.index = 2
                else:
                    self.val += paths
                    self.index = 2
                    return
            elif self.index == 2 and len(self.test) < self.test_len:
                # we certaintly change self.test, but how ?
                if len(paths) > self.test_len - len(self.test):
                    # we have to split paths, because it will fill self.test
                    paths_aux = paths[0:self.test_len - len(self.test)]
                    paths = paths[self.test_len - len(self.test):]
                    self.test += paths_aux
                    self.index = 0
                else:
                    self.test += paths
                    self.index = 0
                    return
            else:
                self.index = (self.index + 1) % 3

    def full(self):
        '''
        '''
        result = (len(self.train) == self.train_len) and(len(self.val) == self.val_len) and (len(self.test) == self.test_len)
        return result
    
    def add_augmented(self):
        '''
        '''
        # we have to grab the original source images of the sets, to add the corresponing augmented images
        # because we can't have for example "image 1.png" in the validation set, and in the train set
        # the image augmented_1_image 1.png

        print("Adding augmented images...")

        aug_image_form = get_augmented_image_name("").split("_")

        # train
        for image_path in self.train:
            image_name, _ = os.path.basename(image_path).split(".")
            aug_corresponding = glob.glob(os.path.join(AUGMENTED_IMAGES_DATA_DIRECTORY, f"{aug_image_form[0]}_*_{image_name}{aug_image_form[2]}"))
            self.train += aug_corresponding

            for aug_img in aug_corresponding:
                with open(TRAIN_TXT_DIRECTORY, 'a' ) as f:
                    f.write(f"{aug_img}\n")

        # val
        for image_path in self.val:
            image_name, _ = os.path.basename(image_path).split(".")
            aug_corresponding = glob.glob(os.path.join(AUGMENTED_IMAGES_DATA_DIRECTORY, f"{aug_image_form[0]}_*_{image_name}{aug_image_form[2]}"))
            self.val += aug_corresponding

            for aug_img in aug_corresponding:
                with open(VALIDATION_TXT_DIRECTORY, 'a' ) as f:
                    f.write(f"{aug_img}\n")
        
        # test
        for image_path in self.test:
            image_name, _ = os.path.basename(image_path).split(".")
            aug_corresponding = glob.glob(os.path.join(AUGMENTED_IMAGES_DATA_DIRECTORY, f"{aug_image_form[0]}_*_{image_name}{aug_image_form[2]}"))
            self.test += aug_corresponding

            for aug_img in aug_corresponding:
                with open(TEST_TXT_DIRECTORY, 'a' ) as f:
                    f.write(f"{aug_img}\n")
            
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


def divide_data_sets(k: int, train_len: int, val_len: int, test_len: int) -> None:
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
    
    for cluster_id in range(0, k):  
        sets_division.add(clusters[cluster_id])

    print("Divided the default + added dataset into:")
    print("- Training set:", len(sets_division.train))
    print("- Validation set:", len(sets_division.val))
    print("- Test set:", len(sets_division.test), "\n")

    sets_division.add_augmented()

    print("Final datasets division:")
    print("- Training set:", len(sets_division.train))
    print("- Validation set:", len(sets_division.val))
    print("- Test set:", len(sets_division.test), "\n")

    sets_division.save()

    print(f"Saved augmented image paths in {TRAIN_TXT_DIRECTORY}")


divide_data_sets(k=20, train_len=800, val_len=328, test_len=0)