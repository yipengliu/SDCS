from dataset import dataset_full
import torch
from torch.autograd import Variable
from torch import nn
import os
import numpy as np
import glob
from utils import *
from scipy import io


class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64,64,3,padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)


    def forward(self, inputs):
        # output=inputs-output
        outputs = torch.relu(inputs+self.conv1(inputs))
        outputs = self.conv2(outputs)
        return outputs

class Reconstruction_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,64,3,padding=1)
        self.residualBlocks = nn.Sequential(ResidualBlock(),
                                             ResidualBlock(),
                                             ResidualBlock(),
                                             ResidualBlock(),
                                             ResidualBlock())
        self.conv2 = nn.Conv2d(64,1,3,padding=1)

    def forward(self, inputs):
        # output=inputs-output
        outputs = torch.relu(self.conv1(inputs))
        outputs = self.residualBlocks(outputs)
        outputs = self.conv2(outputs)
        return outputs


class CSNet(nn.Module):
    def __init__(self,A):
        super().__init__()
        self.register_parameter("A", nn.Parameter(torch.from_numpy(A).float(), requires_grad=True))
        self.register_parameter("Q", nn.Parameter(torch.transpose(torch.from_numpy(A).float(),0,1), requires_grad=True))
        self.reconstruction_model = Reconstruction_model()

    def forward(self, inputs, sampling_matrix_mask):
        # all the inputs are the same
        # inputs = torch.transpose(torch.reshape(torch.squeeze(inputs),[-1,33*33]),0,1)
        H = int(inputs.shape[1] / 33)
        L = int(inputs.shape[2] / 33)
        S = inputs.shape[0]

        now_mask = self.A * sampling_matrix_mask

        now_Q = torch.transpose(sampling_matrix_mask, 1, 2) * self.Q
        now_Q = torch.unsqueeze(now_Q,dim=1)
        # inputs = torch.transpose(inputs,0,1)
        y = self.sampling(now_mask, inputs)  # sampling
        X = torch.matmul(now_Q.expand(S,y.shape[1],-1,-1), y)

        outputs = torch.squeeze(X,dim=3)
        outputs = self.together(outputs, S, H, L)
        # outputs = torch.unsqueeze(torch.reshape(outputs, [-1, 33, 33]), dim=1)
        outputs = outputs+self.reconstruction_model(outputs)
        # outputs = torch.transpose(torch.reshape(torch.squeeze(outputs),[-1,33*33]),0,1)

        return torch.squeeze(outputs,dim=1)


    def sampling(self,A,inputs):
        # inputs = torch.squeeze(inputs)
        # inputs = torch.reshape(inputs,[-1,33*33])  # 矩阵向量hua
        S = inputs.shape[0]
        H = int(inputs.shape[1]/33)
        L = int(inputs.shape[2]/33)
        inputs = torch.unsqueeze(inputs,1)
        inputs = torch.cat(torch.split(inputs,split_size_or_sections=33,dim=2),dim=1)  # H 方向
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=33, dim=3), dim=1)  # L 方向
        inputs = torch.reshape(inputs, [S,H*L,33*33])
        inputs = torch.unsqueeze(inputs,dim=3)
        A_temp = torch.unsqueeze(A,dim=1)
        A_temp = A_temp.expand([-1,inputs.shape[1], -1, -1])
        outputs = torch.matmul(A_temp, inputs)
        return outputs

    def together(self,inputs,S,H,L):
        inputs = torch.reshape(inputs,[S,H*L,33,33])
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=H, dim=1), dim=3)
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=1, dim=1), dim=2)
        return inputs

def get_mask(data_batch,test=0):
    data = torch.zeros([data_batch,math.ceil(1089*0.5),1089])
    for n in range(data_batch):
        if test==0:
            random_num = math.ceil(1089*(random.randint(1,50)/100))
        else:
            random_num = math.ceil(1089*(test/100))
        data[n,0:random_num,:] = 1
    return data


def compute_loss(outputs,target):
    loss = []
    for output in outputs:
        loss.append(torch.mean((output-target)**2))
    return loss

def get_final_loss(loss_all):
    output = 0
    for loss in loss_all:
        output += loss
    return output

def train(model,optG,train_loader,epoch,batch_size,CS_ratio):
    model.train()
    n = 0

    real_label = 1
    fake_label = 0
    criterion = nn.BCELoss().cuda()
    for data in train_loader:
        n = n + 1
        batch_size_ = data.shape[0]
        data = Variable(data.float().cuda())
        # G 训练两次， D 训练一次

        data_batch = batch_size_
        sampling_matrix_mask = get_mask(data_batch)
        sampling_matrix_mask = Variable(sampling_matrix_mask.float().cuda())

        # optD.zero_grad()
        # output = D(data)
        # label = torch.FloatTensor(batch_size_).cuda()
        # label.data.resize_(batch_size_).fill_(real_label)
        # errD_real = criterion(output, label)
        # errD_real.backward()
        # D_x = output.data.mean()

        # fake = model(data,sampling_matrix_mask)
        # label.data.fill_(fake_label)
        # output = D(fake.detach())
        # errD_fake = criterion(output, label)
        # errD_fake.backward()
        # D_G_z1 = output.data.mean()
        # errD = errD_real + errD_fake  # 这两个还是要输出，以便观察
        # optD.step()
        # for _ in  range(2):
        optG.zero_grad()  # 清空梯度
        fake = model(data, sampling_matrix_mask)
        # label.data.fill_(real_label)
        # output = D(fake)
        # D_G_z2 = output.data.mean()
        # errG_D = criterion(output, label)  # 这两个还是要输出，以便观察
        err_l2 = torch.mean((fake-data)**2)
        # loss_all = compute_loss(outputs,data)
        # loss = get_final_loss(loss_all)
        loss = err_l2
        loss.backward()
        optG.step()
        if n % 25 == 0:
            output = "CS_ratio: %d [%02d/%02d] loss: %.4f" % (CS_ratio, epoch, batch_size * n, loss.data.item())
            # output = "[%02d/%02d] cost: %.4f, cost_sym: %.4f \n" % (epoch, batch_size*n,
            #                                        cost.data.item(),cost_sym.data.item())
            print(output)

def get_val_result(model,is_cuda=True):
    model.eval()
    val_CS_ratios = [50, 40, 30, 25, 10, 4, 1]
    test_set_path = "../../dataset/BSR_bsds500/BSR/BSDS500/data/images/val"
    test_set_path = glob.glob(test_set_path + '/*.tif')
    ImgNum = len(test_set_path)  # 测试图像的数量
    PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
    PSNR_CS_ratios = np.zeros([1, len(val_CS_ratios)], dtype=np.float32)
    model.eval()
    n=0
    for CS_ratio in val_CS_ratios:
        for img_no in range(ImgNum):
            imgName = test_set_path[img_no]  # 当前图像的名字

            [Iorg, row, col, Ipad, row_new, col_new] = imread_CS_py(imgName)
            Icol = img2col_py(Ipad, 33) / 255.0  # 返回 行向量化后的图像数据
            # Img_input = np.dot(Icol, Phi_input)  # 压缩感知降采样
            # Img_output = Icol
            if is_cuda:
                inputs = Variable(torch.from_numpy(Ipad.astype('float32')/255.0).cuda())
            else:
                inputs = Variable(torch.from_numpy(Ipad.astype('float32')/255.0))
            # if model.network == "ista_plus" or model.network == "ista":
            #     output, _ = model(inputs)
            # else:
            #     output = model(inputs)
            inputs = torch.unsqueeze(inputs, dim=0)
            sampling_matrix_mask = get_mask(inputs.shape[0], CS_ratio)
            sampling_matrix_mask = Variable(sampling_matrix_mask.float().cuda())
            output = model(inputs,sampling_matrix_mask)
            output = torch.squeeze(output)
            if is_cuda:
                output = output.cpu().data.numpy()
            else:
                output = output.data.numpy()
            images_recovered = output[0:row,0:col]
            rec_PSNR = psnr(images_recovered * 255, Iorg)  # 计算PSNR的值
            PSNR_All[0, img_no] = rec_PSNR
        PSNR_CS_ratios[0, n] = np.mean(PSNR_All)
        n += 1

    return PSNR_CS_ratios


def load_sampling_matrix(CS_ratio):
    path = "../../dataset/sampling_matrix"
    data = io.loadmat(os.path.join(path, str(CS_ratio) + '.mat'))['sampling_matrix']
    return data



if __name__ == "__main__":
    is_cuda = True
    CS_ratio = 25  # 4, 10, 25, 30, 40, 50
    # n_output = 1089
    SNRs = [40,30,25,15]
    CS_ratios = [50,40,30,25,10,4,1]  # block 数目为 5
    # nrtrain = 88912
    learning_rate_G = 0.0001
    learning_rate_D = 0.00001
    EpochNum = 100
    batch_size = 64
    results_saving_path = "../../results_c4/CSNet"

    if not os.path.exists(results_saving_path):
        os.mkdir(results_saving_path)

    # results_saving_path = os.path.join(results_saving_path,"ReconNet_adv_nomask_nodeblock")
    if not os.path.exists(results_saving_path):
        os.mkdir(results_saving_path)

    print('Load Data...')  # jiazaishuju

    train_dataset = dataset_full(root="../../dataset", train=True, transform=None,
                            target_transform=None)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=2)
    for CS_ratio in CS_ratios:
        A = load_sampling_matrix(CS_ratio)
        model = CSNet(A)  # 生成器
        optG = torch.optim.Adam(model.parameters(), lr=learning_rate_G)
        model.cuda()

        sub_path = os.path.join(results_saving_path, str(CS_ratio))

        if not os.path.exists(sub_path):
            os.mkdir(sub_path)
        best_psnr = 0
        for epoch in range(1,EpochNum+1):
            if epoch==101:
                optG.defaults['lr'] *= 0.2

            train(model,optG,train_loader,epoch,batch_size,CS_ratio)
            psnr_cs_ratios = get_val_result(model)
            mean_psnr = np.mean(psnr_cs_ratios)

            print_str = "epoch: %d  psnr: mean %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f" % (
            epoch, mean_psnr, psnr_cs_ratios[0, 0], psnr_cs_ratios[0, 1], psnr_cs_ratios[0, 2],
            psnr_cs_ratios[0, 3], psnr_cs_ratios[0, 4], psnr_cs_ratios[0, 5], psnr_cs_ratios[0, 6])
            print(print_str)

            output_file = open(sub_path + "/log_PSNR.txt", 'a')
            output_file.write("PSNR: %.4f\n" % (mean_psnr))
            output_file.close()

            if mean_psnr>best_psnr:
                best_psnr = mean_psnr
                output_file = open(sub_path + "/log_PSNR_best.txt", 'a')
                output_file.write("PSNR: %.4f\n" % (best_psnr))
                output_file.close()
                torch.save(model.state_dict(), sub_path + "/best_model.pkl")
