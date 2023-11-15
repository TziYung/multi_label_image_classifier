import cv2
import os
import random
def image_process(dir_path: str, img_size: tuple) -> list:
    # dir path would be the path of the directory that contains images
    # img_size is the width and length of the image in tuple

    print(f"Loading from {dir_path}")
    processed_images = []
    
    for img_name in tqdm(os.listdir(dir_path)):
        img_path = os.path.join(dir_path, img_name)

        # If the path is not a file(could be a directory), ignore it
        if os.path.isfile(img_path) == False:
            continue

        try:
            # read, resize image, and convert image from bgr to rgb due to the reason
            # that opencv read image in the pattern of bgr
            img = cv2.imread(path)
            img = cv2.resize(img, img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            processed_images.append(img)

        except Exception as e:
            print(e)
            print(f"Can't load image from: {path}")
    return processed_images

class MultiLabelLoader():
    def __init__(self, dir_path: str, img_size: tuple, batch_size: int, split_ratio: tuple):
        self.dir_path = dir_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self.dir_list = self.get_dir_list()        
        self.class_list = self.get_class()

        
    def get_dir_list(self):
        # Get all the directory's name that is contained in self.dir_path
        return [directory for directory in os.listdir(self.dir_path) if
            os.path.isdir(os.path.join(self.dir_path, directory))]
    def get_class(self):
        # Get the name of each class, the name should be a string with no '_' because
        # '_' would be used for directory that two class appear in same image
        return [class_ for class_ in os.listdir(self.dir_path) if class_.count('_') == 0]
    def label_to_onehot(self, labels: list):
        # Convert given label to onehot encoding 
        one_hot = [ 1 if class_ in labels else 0 for class_ in self.class_list]
        
        return one_hot

    def shuffle(self, data ,label):
        # Would create data-label pair then shuffle it
        data_label_pair = list(zip(train_data, train_label))
        random.shuffle(data_label_pair)
        data, label = list(zip(*data_label_pair))
        return np.array(data), np.array(label)
        
    def load_image(self):
        # Loop through all dir and load image init
        data = []
        label = []

        for dir_name in self.dir_list:
            dir_label = self.label_to_onehot(dir_name.split("_")) 
            target_dir_path = os.path.join(self.dir_path, dir_name)
            data.extend(image_process(target_dir_path, self.img_size))
            label.extend([dir_label for _ in range(len(data) - len(label))])
        data, label = self.shuffle(data, label)
        self.split(data, label)
    def __getitem__(self, batch_index: int):
        # Return data and label with given batch index
        start_index = batch_index  * self.batch_size
        end_index = start_index + self.batch_size
        data = np.array(self.train_data[start_index: end_index])
        label = np.array(self.train_label[start_index: end_index])
        # Shuffle it when it is the end of the epoch
        if end_index >= len(self.train_data):
            self.shuffle()
    def __len__(self):
        return int(len(self.train_data)/self.batch_size) + min(len(self.train_data) % self.batch_size, 1)
    # Split the data with given ratio 
    def split(self, data, label):
        train_amount, val_amount, test_amount = [int(ratio * len(data)) for ratio in self.split_ratio]
        self.test_data, self.test_label = data[-test_amount:], label[-test_amount:]
        self.val_data, self.val_label = data[-val_amount: -test_amount], label[-val_amount: -test_amount]
        self.train_data, self.train_label = data[:-val_amount], label[:-val_amount]

