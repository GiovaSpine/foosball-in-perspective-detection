import glob
from collections import defaultdict
from config import *
from utility import *


class SetsDivision:
    def __init__(self, train_len: int, val_len: int, test_len: int):
        '''
        Constructor for the SetsDivision class.

        Parameters:
        train_len (int): The length the train set should have
        val_len (int): The length the val set should have
        test_len (int): The length the test set should have
        '''
        # check parameters
        if not isinstance(train_len, int):
            raise TypeError(f"Warning: train_len must be an integer. {type(train_len)} given")
        if train_len < 0:
            raise ValueError(f"Warning: train_len must be >= 0. {train_len} given")
        
        if not isinstance(val_len, int):
            raise TypeError(f"Warning: val_len must be an integer. {type(val_len)} given")
        if val_len < 0:
            raise ValueError(f"Warning: val_len must be >= 0. {val_len} given")
        
        if not isinstance(test_len, int):
            raise TypeError(f"Warning: test_len must be an integer. {type(test_len)} given")
        if test_len < 0:
            raise ValueError(f"Warning: test_len must be >= 0. {test_len} given")

        self.train = []
        self.val = []
        self.test = []
        self.train_len = train_len
        self.val_len = val_len
        self.test_len = test_len
        self.index = 0
        
    def add(self, cluster: list) -> None:
        '''
        Adds a cluster to one of train, val and test sets.

        Parameters:
        cluster (list): The cluster as a list of image names without extension

        Returns:
        None
        '''
        if not isinstance(cluster, list):
            raise TypeError("Warning: cluster should be a list")
        if not all(isinstance(image, str) for image in cluster):
            raise ValueError("Warning: cluster should contain strings, that are image names")

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
    
    def add_augmented(self) -> None:
        '''
        Adds the augmented images to the sets that contain their source/original image.
        For example: if "image 1.jpg" (source/original image) is in the val set,
        the image "augmented_1_image 1.jpg" (augmented counterpart) has to be added to the
        val set, to avoid contamination (the validation set contains images close to the training ones).
        '''
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
            
    def save(self) -> None:
        '''
        Saves the sets as train.txt, val.txt and test.txt, in a predefined directory,
        each containg image paths of those sets.
        '''
        # train
        with open(TRAIN_TXT_DIRECTORY, 'w') as f:
            for path in self.train,:
                f.write(f"{path}\n")
        print(f"Saved image paths in {TRAIN_TXT_DIRECTORY}")

        # val
        with open(VALIDATION_TXT_DIRECTORY, 'w') as f:
            for path in self.val,:
                f.write(f"{path}\n")
        print(f"Saved image paths in {VALIDATION_TXT_DIRECTORY}")

        # test
        with open(TEST_TXT_DIRECTORY, 'w') as f:
            for path in self.test,:
                f.write(f"{path}\n")
        print(f"Saved image paths in {TEST_TXT_DIRECTORY}")


def divide_dataset(k: int, train_len: int, val_len: int, test_len: int) -> None:
    '''
    Given a k (number of clusters) greater than 3, and a division of the number of images
    of the default + added dataset, asas train_len, val_len, test_len,
    it divides the dataset into a train, val and test set and 
    creates train.txt, val.txt, and test.txt in a predefined directory,
    where the absolute paths of the images will be written and used by yolo.

    Parameters:
    k (int): The clustering analyisis/division done for that k (number of clusters)
    train_len (int): The split of the default + added dataset for the train set
    val_len (int): The split of the default + added dataset for the validation set
    test_len (int): The split of the default + added dataset for the test set

    Returns:
    None
    '''
    # check parameters
    if not isinstance(k, int):
        raise TypeError(f"Warning: k must be an integer. {type(k)} given")
    if k < 3 or k > MAX_N_CLUSTERS:
        raise ValueError(f"Warning: k must between 3 and {MAX_N_CLUSTERS}")
    
    if not isinstance(train_len, int):
        raise TypeError(f"Warning: train_len must be an integer. {type(train_len)} given")
    if train_len < 0:
        raise ValueError(f"Warning: train_len must be >= 0. {train_len} given")
    
    if not isinstance(val_len, int):
        raise TypeError(f"Warning: val_len must be an integer. {type(val_len)} given")
    if val_len < 0:
        raise ValueError(f"Warning: val_len must be >= 0. {val_len} given")
    
    if not isinstance(test_len, int):
        raise TypeError(f"Warning: test_len must be an integer. {type(test_len)} given")
    if test_len < 0:
        raise ValueError(f"Warning: test_len must be >= 0. {test_len} given")

    # load the clustering (we need to know in which cluster each image belongs)
    all_clustering_labels = load_all_clustering_label(ADDED_CLUSTERING_DIRECTORY)

    # dictionary to group images (cluster_id -> [list of image names])
    clusters = defaultdict(list)
    for img_name, values in all_clustering_labels.items():
        # we have to group based on the value of the element of index (k - MIN_N_CLUSTERS)
        # of the labels list
        index = k - MIN_N_CLUSTERS
        key = values[index]
        clusters[key].append(img_name)

    # object to divide the dataset
    sets_division = SetsDivision(train_len, val_len, test_len)
    
    # let's assign each cluster for a specific set
    # the idea is that different clusters contain different images
    # and so the validation set for example, shouldn't contain
    # images too similar to the ones of the training set
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


divide_dataset(k=20, train_len=800, val_len=328, test_len=0)