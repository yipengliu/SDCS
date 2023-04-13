import torch
import os
from scipy import io
import random


class dataset():
    """
    加载数据，并进行预处理后保存
    """
    training_file = "train_data.pt"  # 训练数据文件
    def __init__(self,root='dataset', train=True, transform=None,target_transform=None):
        """
        :param root: 数据存放位置 默认存放在当前路径下
        :param train: 是否加载的是
        :param transform: 经过的变换
        """
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        # 现在检查是否存在原始文件


        # 若原始文件存在，检查是否有 .pt文件
        if not (os.path.exists(os.path.join(self.root,self.training_file))):

            # 加载文件,并将文件中的数据保存为 .pt 文件
            self.generate_train_data()

        if train:  # 如果加载训练集
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.training_file))

    def __len__(self):
        """
        获得数据集 样本数量
        :return:  数量
        """
        if self.train:
            return len(self.train_data)

    def __getitem__(self, index):
        """
        获得数据集得中得一个数据
        :param index:  坐标
        :return:
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img.numpy(), mode="RGB")
        # target = Image.fromarray(target.numpy(), mode="RGB")

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.train:
            return img, target
        else:
            return img, target

    def generate_train_data(self,path="dataset"):
        """
        该函数就是负责生成训练集
        :param path:
        :return:
        """
        all_path = os.path.join(path,"BSR_bsds500/BSR/BSDS500/data/images/train")
        inputs = []
        labels = []
        n = 0
        for roots,dirs,files in os.walk(all_path):
            if roots==all_path:
                for file in files:
                    n+=1
                    print(n)
                    if file[-4:] == ".mat":
                        data = io.loadmat(os.path.join(all_path, file))['data']  # 这就是数据
                        sub_inputs,sub_labels = self.get_data_random(data,977)
                        inputs.extend(sub_inputs)
                        labels.extend(sub_labels)
        train_inputs = torch.Tensor(inputs)
        train_labels = torch.Tensor(labels)
        training_set = (train_inputs, train_labels)

        with open(os.path.join(path, "train_data.pt"), 'wb') as f:
            torch.save(training_set, f)
        # with open(os.path.join(self.root, self.dir_name, self.testing_file), 'wb') as f:
        #     torch.save(test_set, f)
        print("文件已经打包为 .pt 文件")
    def get_data_random(self,data, num):
        im_size = 33
        h = data.shape[0]
        l = data.shape[1]
        inputs = []
        labels = []
        for n in range(num):
            x = random.randint(0, h - im_size)
            y = random.randint(0, l - im_size)
            sub_im = data[x:x + im_size, y:y + im_size]
            inputs.append(sub_im)
            labels.append(sub_im)
        return inputs, labels


class dataset_full():
    """
    加载数据，并进行预处理后保存
    """
    training_file = "train_data_full.pt"  # 训练数据文件
    def __init__(self,root='dataset', train=True, transform=None,target_transform=None):
        """
        :param root: 数据存放位置 默认存放在当前路径下
        :param train: 是否加载的是
        :param transform: 经过的变换
        """
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        # 现在检查是否存在原始文件


        # 若原始文件存在，检查是否有 .pt文件
        if not (os.path.exists(os.path.join(self.root,self.training_file))):

            # 加载文件,并将文件中的数据保存为 .pt 文件
            self.generate_train_data()

        if train:  # 如果加载训练集
            self.train_data = torch.load(
                os.path.join(self.root, self.training_file))[0]

    def __len__(self):
        """
        获得数据集 样本数量
        :return:  数量
        """
        if self.train:
            return len(self.train_data)

    def __getitem__(self, index):
        """
        获得数据集得中得一个数据
        :param index:  坐标
        :return:
        """
        if self.train:
            img = self.train_data[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img.numpy(), mode="RGB")
        # target = Image.fromarray(target.numpy(), mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.train:
            return img
        else:
            return img

    def generate_train_data(self, path="dataset"):
        """
        该函数就是负责生成训练集
        :param path:
        :return:
        """
        all_path = os.path.join(path, "BSR_bsds500/BSR/BSDS500/data/images/train")
        inputs = []
        n = 0
        for roots, dirs, files in os.walk(all_path):
            if roots == all_path:
                for file in files:
                    n += 1
                    print(n)
                    if file[-4:] == ".mat":
                        data = io.loadmat(os.path.join(all_path, file))['data']  # 这就是数据

                        sub_inputs, sub_labels = self.get_data_random(data, 448)
                        inputs.extend(sub_inputs)
                        # labels.extend(sub_labels)
        train_inputs = torch.Tensor(inputs)
        # train_labels = torch.Tensor(labels)
        training_set = (train_inputs,)

        with open(os.path.join(path, "train_data_full.pt"), 'wb') as f:
            torch.save(training_set, f)
        # with open(os.path.join(self.root, self.dir_name, self.testing_file), 'wb') as f:
        #     torch.save(test_set, f)
        print("文件已经打包为 .pt 文件")

    def get_data_random(self,data, num):
        im_size = 33*3
        h = data.shape[0]
        l = data.shape[1]
        inputs = []
        labels = []
        for n in range(num):
            x = random.randint(0, h - im_size)
            y = random.randint(0, l - im_size)
            sub_im = data[x:x + im_size, y:y + im_size]
            inputs.append(sub_im)
            labels.append(sub_im)
        return inputs, labels

if __name__ == "__main__":
    dataset_full()