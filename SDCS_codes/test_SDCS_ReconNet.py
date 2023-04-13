from dataset import dataset
import torch
from torch.autograd import Variable
from torch import nn
import os
import numpy as np
import glob
from utils import *
from scipy import io


class ReconNet(nn.Module):
    def __init__(self,A):
        super().__init__()
        self.register_parameter("A", nn.Parameter(torch.from_numpy(A).float(), requires_grad=True))
        self.register_parameter("Q", nn.Parameter(torch.transpose(torch.from_numpy(A).float(),0,1), requires_grad=True))
        self.conv1 = nn.Conv2d(1,64,11,padding=5)
        self.conv2 = nn.Conv2d(64,32,1,padding=0)
        self.conv3 = nn.Conv2d(32,1,7,padding=3)
        self.conv4 = nn.Conv2d(1, 64, 11, padding=5)
        self.conv5 = nn.Conv2d(64, 32, 1, padding=0)
        self.conv6 = nn.Conv2d(32, 1, 7, padding=3)

    def forward(self, inputs, sampling_matrix_mask):
        # all the inputs are the same
        now_mask = self.A * sampling_matrix_mask

        now_Q = torch.transpose(sampling_matrix_mask, 1, 2) * self.Q
        # inputs = torch.transpose(inputs,0,1)
        y = self.sampling(now_mask, inputs)  # sampling
        X = torch.matmul(now_Q, y)

        outputs = torch.squeeze(X)
        outputs = torch.unsqueeze(torch.reshape(outputs, [-1, 33, 33]), dim=1)
        outputs = torch.relu(self.conv1(outputs))
        outputs = torch.relu(self.conv2(outputs))
        outputs = torch.relu(self.conv3(outputs))
        outputs = torch.relu(self.conv4(outputs))
        outputs = torch.relu(self.conv5(outputs))
        outputs = self.conv6(outputs)
        outputs = torch.transpose(torch.reshape(torch.squeeze(outputs),[-1,33*33]),0,1)

        return outputs


    def sampling(self,A,inputs):
        # inputs = torch.squeeze(inputs)
        # inputs = torch.reshape(inputs,[-1,33*33])  # 矩阵向量hua
        inputs = torch.transpose(inputs, 0, 1)
        inputs = torch.unsqueeze(inputs, dim=2)
        outputs = torch.matmul(A, inputs)
        return outputs

class Discr(nn.Module):
    def __init__(self):
        super(Discr, self).__init__()
        ndf = 16
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            # nn.Conv2d(1, ndf, 4, 2, 1, bias=False),
            # nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(1, 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(4, 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(4, 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        outputs = torch.transpose(input, 0, 1)
        outputs = torch.unsqueeze(torch.reshape(outputs, [-1, 33, 33]), dim=1)
        outputs = self.main(outputs)

        return outputs.view(-1, 1)


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

def train(model,D,optG,optD,train_loader,epoch,batch_size,CS_ratio):
    model.train()
    n = 0

    real_label = 1
    fake_label = 0
    criterion = nn.BCELoss().cuda()
    for data, target in train_loader:
        n = n + 1
        batch_size_ = data.shape[0]
        data, target = torch.transpose(torch.reshape(data,[-1,33*33]),0,1),torch.transpose(torch.reshape(target,[-1,33*33]),0,1)
        data, target = Variable(data.float().cuda()), Variable(target.float().cuda())
        # G 训练两次， D 训练一次

        optD.zero_grad()
        output = D(data)
        label = torch.FloatTensor(batch_size_).cuda()
        label.data.resize_(batch_size_).fill_(real_label)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.data.mean()

        fake = model(data)
        label.data.fill_(fake_label)
        output = D(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake  # 这两个还是要输出，以便观察
        optD.step()
        for _ in  range(2):
            optG.zero_grad()  # 清空梯度
            fake = model(data)
            label.data.fill_(real_label)
            output = D(fake)
            D_G_z2 = output.data.mean()
            errG_D = criterion(output, label)  # 这两个还是要输出，以便观察
            err_l2 = torch.mean((fake-target)**2)
            # loss_all = compute_loss(outputs,data)
            # loss = get_final_loss(loss_all)
            loss = 0.00001 * errG_D + err_l2
            loss.backward()
            optG.step()
        if n % 25 == 0:
            output = "CS_ratio: %d [%02d/%02d] cost: %.4f  D_1: %.4f  D_2: %.4f  D_3: %.4f" % (CS_ratio,epoch, batch_size * n, err_l2.data.item(),D_x.data.item(),D_G_z1.data.item(),D_G_z2.data.item())
            # output = "[%02d/%02d] cost: %.4f, cost_sym: %.4f \n" % (epoch, batch_size*n,
            #                                        cost.data.item(),cost_sym.data.item())
            print(output)


def get_mask(data_batch,test=0):
    data = torch.zeros([data_batch,math.ceil(1089*0.5),1089])
    for n in range(data_batch):
        if test==0:
            random_num = math.ceil(1089*(random.randint(1,50)/100))
        else:
            random_num = math.ceil(1089*(test/100))
        data[n,0:random_num,:] = 1
    return data


def get_val_result(model,is_cuda=True):
    model.eval()
    val_CS_ratios = [50, 40, 30, 25, 10, 4, 1]
    with torch.no_grad():
        test_set_path = "../../dataset/Set11"
        test_set_path = glob.glob(test_set_path + '/*.tif')
        ImgNum = len(test_set_path)  # 测试图像的数量
        PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
        SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)
        PSNR_CS_ratios = np.zeros([1, 50], dtype=np.float32)
        SSIM_CS_ratios = np.zeros([1, 50], dtype=np.float32)
        sub_path = "../../results_c4/ReconNet"

        n = 0
        for CS_ratio in val_CS_ratios:
            print(n + 1)
            for img_no in range(ImgNum):
                imgName = test_set_path[img_no]  # 当前图像的名字

                [Iorg, row, col, Ipad, row_new, col_new] = imread_CS_py(imgName)
                Icol = img2col_py(Ipad, 33) / 255.0  # 返回 行向量化后的图像数据
                Ipad /= 255.0
                # Img_input = np.dot(Icol, Phi_input)  # 压缩感知降采样
                # Img_output = Icol

                if is_cuda:
                    inputs = Variable(torch.from_numpy(Icol.astype('float32')).cuda())
                else:
                    inputs = Variable(torch.from_numpy(Icol.astype('float32')))
                # if model.network == "ista_plus" or model.network == "ista":
                #     output, _ = model(inputs)
                # else:
                #     output = model(inputs)
                # inputs = torch.unsqueeze(torch.unsqueeze(inputs, dim=0), dim=0)
                sampling_matrix_mask = get_mask(inputs.shape[1], CS_ratio)

                if is_cuda:
                    sampling_matrix_mask = Variable(sampling_matrix_mask.float().cuda())
                else:
                    sampling_matrix_mask = Variable(sampling_matrix_mask.float())

                # inputs = torch.transpose(inputs,0,1)

                outputs = model(inputs, sampling_matrix_mask)
                # outputs = torch.transpose(outputs, 0, 1)
                if is_cuda:
                    outputs = outputs.cpu().data.numpy()
                else:
                    outputs = outputs.data.numpy()

                images_recovered = col2im_CS_py(outputs, row, col, row_new, col_new)


                images_recovered = images_recovered * 255

                # images_recovered = BM3D_denoise(images_recovered,Iorg,model.detach().A)
                # images_recovered *= 255
                aaa = images_recovered.astype(int)
                bbb = aaa < 0
                aaa[bbb] = 0
                bbb = aaa > 255
                aaa[bbb] = 255

                rec_PSNR = psnr(aaa, Iorg)  # 计算PSNR的值
                PSNR_All[0, img_no] = rec_PSNR
                rec_SSIM = compute_ssim(aaa, Iorg)  # 计算PSNR的值
                SSIM_All[0, img_no] = rec_SSIM

            PSNR_CS_ratios[0, CS_ratio-1] = np.mean(PSNR_All)
            SSIM_CS_ratios[0, CS_ratio-1] = np.mean(SSIM_All)
            output_file = open(sub_path + "/Set11_PSNR_1_to_50.txt", 'a')
            output_file.write("%.4f " % (np.mean(PSNR_All)))
            output_file.close()
            output_file = open(sub_path + "/Set11_SSIM_1_to_50.txt", 'a')
            output_file.write("%.4f " % (np.mean(SSIM_All)))
            output_file.close()
            n += 1
    return PSNR_CS_ratios, SSIM_CS_ratios


def load_sampling_matrix(CS_ratio):
    path = "../../dataset/sampling_matrix"
    data = io.loadmat(os.path.join(path, str(CS_ratio) + '.mat'))['sampling_matrix']
    return data



if __name__ == "__main__":

    model_name = "ReconNet"

    CS_ratio = 50
    phase = 9

    path = os.path.join("../../results", model_name, str(CS_ratio), "best_model.pkl")
    A = load_sampling_matrix(CS_ratio)
    model = ReconNet(A)
    model.cuda()
    model.load_state_dict(torch.load(path))
    print("Start")
    psnrs, ssims = get_val_result(model, is_cuda=True)  # test AMP_net

    print_str = " %.4f %.4f %.4f %.4f %.4f %.4f %.4f" % (
        psnrs[0, 49], psnrs[0, 39], psnrs[0, 29],
        psnrs[0, 24], psnrs[0, 9], psnrs[0, 3], psnrs[0, 0])
    print(print_str)

    print_str = " %.4f %.4f %.4f %.4f %.4f %.4f %.4f" % (
        ssims[0, 49], ssims[0, 39], ssims[0, 29],
        ssims[0, 24], ssims[0, 9], ssims[0, 3], ssims[0, 0])
    print(print_str)
