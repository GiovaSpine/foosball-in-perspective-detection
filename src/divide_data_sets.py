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
    Given a k (number of clusters) greater than 3, and the number of images each set
    should have (training, validation, test), it produces train.txt, val.txt, test.txt,
    by taking one image at a time from each cluster, in the hope to have different images
    in each set.
    The files train.txt, val.txt, test.txt will contain paths of the images, and will be
    used by yolo.

    Parameters:

    Returns:
    None
    '''

    all_clustering_labels = load_all_clustering_label(AUGMENTED_CLUSTERING_DIRECTORY)

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
                image_path = find_image_path(image_name, AUGMENTED_IMAGES_DATA_DIRECTORY)
                sets_division.add(image_path)

        condition = not sets_division.full()

    print("Divided the sets into:")
    print("- Training set:", len(sets_division.train))
    print("- Validation set:", len(sets_division.val))
    print("- Test set:", len(sets_division.test))
    
    sets_division.save()


divide_training_set(k=18, train_len=2967, val_len=1484, test_len=1483)