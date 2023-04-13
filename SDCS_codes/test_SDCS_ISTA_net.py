from dataset import dataset
import torch
from torch.autograd import Variable
from torch import nn
import os
import numpy as np
import glob
from utils import *
from scipy import io
from skimage.io import imsave


class ISTA_net_f(nn.Module):  # 模型
    # 用于构造 ISTA net 神经网络的那一部分
    def __init__(self,network_name):
        super().__init__()
        """
        构建最核心的网络
        """

        conv_size = 32
        filter_size = 3
        self.network = network_name
        self.register_parameter("soft_thr", nn.Parameter(torch.tensor(0.1)))  # 自己注册一个变量

        self.conv1 = nn.Conv2d(1,conv_size,filter_size,bias=False,padding=1)
        self.conv2 = nn.Conv2d(conv_size,conv_size,filter_size,bias=False,padding=1)
        self.conv3 = nn.Conv2d(conv_size, conv_size, filter_size, bias=False,padding=1)
        self.conv4 = nn.Conv2d(conv_size, conv_size, filter_size, bias=False,padding=1)
        self.conv5 = nn.Conv2d(conv_size, conv_size, filter_size, bias=False,padding=1)
        self.conv6 = nn.Conv2d(conv_size, 1, filter_size, bias=False,padding=1)

    def forward(self,input):
        input = torch.unsqueeze(torch.reshape(torch.transpose(input, 0, 1), [-1, 33, 33]), dim=1)
        if self.network == "ISTA_net":
            x = torch.relu(self.conv2(self.conv1(input)))
            x = self.conv3(x)

            x = torch.mul(torch.sign(x),torch.relu(torch.abs(x)-self.soft_thr))
            x = torch.relu(self.conv4(x))
            x = self.conv6(self.conv5(x))  # 前向传播

            cost = torch.relu(self.conv2(self.conv1(input)))
            cost = self.conv3(cost)
            cost = torch.relu(self.conv4(cost))
            cost = self.conv6(self.conv5(cost))
            cost = cost - input
            output = x
            output = torch.transpose(torch.reshape(output, [-1, 33 * 33]), 0, 1)
            cost = torch.transpose(torch.reshape(cost, [-1, 33 * 33]), 0, 1)

        elif self.network == "ISTA_net_plus":
            x1 = self.conv1(input)
            x1_1 = torch.relu(self.conv2(x1))
            x2 = self.conv3(x1_1)
            x3 = torch.mul(torch.sign(x2), torch.relu(torch.abs(x2) - self.soft_thr))
            x4 = torch.relu(self.conv4(x3))
            x5 = self.conv6(self.conv5(x4))  # 前向传播

            cost = torch.relu(self.conv2(x1))
            cost = self.conv3(cost)
            cost = torch.relu(self.conv4(cost))
            cost = self.conv5(cost)
            cost = cost - x1
            output = x5 + input
            output = torch.transpose(torch.reshape(output, [-1, 33 * 33]), 0, 1)
            cost = torch.transpose(torch.reshape(cost, [-1, 33 * 33]), 0, 1)

        return output, cost


class ISTA_net(nn.Module):
    def __init__(self,layer_num, A,network_name="ISTA_net_plus"):
        super().__init__()
        self.layer_num = layer_num
        self.network_name = network_name
        self.fs = []
        self.steps = []
        self.register_parameter("A",nn.Parameter(torch.from_numpy(A).float(),requires_grad=True))
        self.register_parameter("Q", nn.Parameter(torch.from_numpy(np.transpose(A)).float(), requires_grad=True))

        for n in range(layer_num):
            self.fs.append(ISTA_net_f(network_name))
            self.register_parameter("step_"+str(n+1), nn.Parameter(torch.tensor(0.1)))
            self.steps.append(eval("self.step_"+str(n+1)))

        for n,f in enumerate(self.fs):
            self.add_module("f_"+str(n+1),f)

    def forward(self, inputs, sampling_matrix_mask, output_layers):
        """
        此处就是前向传播，返回每一层的输出
        :param inputs: 此处的inputs 为图像数据
        :return:
        """
        outputs = []
        costs = []
        now_mask = self.A * sampling_matrix_mask
        now_Q = torch.transpose(sampling_matrix_mask, 1, 2) * self.Q
        y = self.sampling(now_mask, inputs)
        X = torch.matmul(now_Q, y)

        for n in range(output_layers):
            step = self.steps[n]
            f = self.fs[n]

            temp = self.block1(now_mask,X,y,step)
            temp = torch.squeeze(temp)
            temp = torch.transpose(temp,0,1)
            X,cost = f(temp)

            outputs.append(X.clone())
            costs.append(cost.clone())
            temp = torch.transpose(temp, 0, 1)
            X = torch.unsqueeze(temp,dim=2)
        return outputs,costs


    def sampling(self,A, inputs):
        # inputs = torch.squeeze(inputs)
        # inputs = torch.reshape(inputs,[-1,33*33])  # 矩阵向量hua
        inputs = torch.transpose(inputs,0,1)
        inputs = torch.unsqueeze(inputs,dim=2)
        outputs = torch.matmul(A, inputs)
        return outputs

    def block1(self,A,X,y,step):
        # X = torch.squeeze(X)
        # X = torch.transpose(torch.reshape(X, [-1, 33 * 33]),0,1)  # 矩阵向量hua
        outputs = step*torch.matmul(torch.transpose(A,1,2),y-torch.matmul(A,X))
        outputs = outputs + X
        # outputs = torch.unsqueeze(torch.reshape(torch.transpose(outputs,0,1),[-1,33,33]),dim=1)
        return outputs


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
        sub_path = "../../results_c4/ISTA_Net"

        n = 0
        for CS_ratio in val_CS_ratios:
            save_path = sub_path + "/images_generated"
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            save_path = os.path.join(save_path, str(CS_ratio))
            if not os.path.exists(save_path):
                os.mkdir(save_path)
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

                outputs,_ = model(inputs, sampling_matrix_mask,9)
                outputs = outputs[-1]
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

                name_temp = (imgName.split('/')[-1]).split('.')[0]
                imsave(os.path.join(save_path, name_temp + '.jpg'), aaa)
                imsave(os.path.join(save_path, name_temp + '_' + str(rec_PSNR) + '_' + str(rec_SSIM) + '.jpg'), aaa)

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

    model_name = "ISTA_Net"

    CS_ratio = 50
    phase = 9

    path = os.path.join("../../results_c4", model_name, str(CS_ratio), str(9), "best_model.pkl")
    A = load_sampling_matrix(CS_ratio)
    model = ISTA_net(9,A)
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
