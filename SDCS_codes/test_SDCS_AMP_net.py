from dataset import dataset_full,dataset
import os
import numpy as np
import glob
from utils import *
from scipy import io
import torch
from torch.nn import Module
from torch import nn
from torch.autograd import Variable
from skimage.io import imsave

"""
No mask training, no deblocking

"""

class Denoiser(Module):
    def __init__(self):
        super().__init__()
        self.D = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1),

                               nn.ReLU(),
                               nn.Conv2d(32, 32, 3, padding=1),

                               nn.ReLU(),
                               nn.Conv2d(32, 32, 3, padding=1),

                               nn.ReLU(),
                               nn.Conv2d(32, 1, 3, padding=1,bias=False))

    def forward(self, inputs):
        S = inputs.shape[0]
        inputs = torch.squeeze(inputs,dim=3)
        inputs = torch.reshape(inputs,[inputs.shape[0],inputs.shape[1],33, 33])

        inputs = torch.cat(torch.split(inputs, split_size_or_sections=1,dim=1),dim=0)

        # inputs = torch.unsqueeze(torch.reshape(inputs,[-1,33,33]),dim=1)
        output = self.D(inputs)
        # output=inputs-output
        output = torch.cat(torch.split(output, split_size_or_sections=S, dim=0), dim=1)
        output = torch.reshape(output,[output.shape[0],output.shape[1],33*33])
        output = torch.unsqueeze(output, dim=3)
        return output

class Deblocker(Module):
    def __init__(self):
        super().__init__()
        self.D = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1),

                               nn.ReLU(),
                               nn.Conv2d(32, 32, 3, padding=1),

                               nn.ReLU(),
                               nn.Conv2d(32, 32, 3, padding=1),

                               nn.ReLU(),
                               nn.Conv2d(32, 1, 3, padding=1,bias=False))

    def forward(self, inputs):
        # inputs = torch.unsqueeze(inputs,dim=1)
        output = self.D(inputs)
        # output = torch.squeeze(output,dim=1)
        return output


class AMP_net_Deblock_RA(Module):
    def __init__(self,layer_num, A):
        super().__init__()
        self.layer_num = layer_num
        self.denoisers = []
        self.deblocks = []
        self.steps = []
        for n in range(layer_num):
            self.denoisers.append(Denoiser())
            self.deblocks.append(Deblocker())
            self.register_parameter("step_" + str(n + 1), nn.Parameter(torch.tensor(1.0),requires_grad=True))
            self.steps.append(eval("self.step_" + str(n + 1)))
        for n,denoiser in enumerate(self.denoisers):
            self.add_module("denoiser_"+str(n+1),denoiser)
        for n,deblock in enumerate(self.deblocks):
            self.add_module("deblock_"+str(n+1),deblock)

        for p in self.parameters():
            p.requires_grad=False

        self.register_parameter("A", nn.Parameter(torch.from_numpy(A).float(), requires_grad=True))
        self.register_parameter("Q", nn.Parameter(torch.from_numpy(np.transpose(A)).float(), requires_grad=True))

    def forward(self, inputs, sampling_matrix_mask, output_layers):
        """
        此处就是前向传播，返回每一层的输出
        :param inputs: 此处的inputs 为图像数据
        :return:
        """
        H = int(inputs.shape[2]/33)
        L = int(inputs.shape[3]/33)
        S = inputs.shape[0]

        now_mask = self.A*sampling_matrix_mask
        now_Q = torch.transpose(sampling_matrix_mask,0,1)*self.Q
        y = self.sampling(now_mask, inputs)  # sampling

        now_Q = torch.unsqueeze(torch.unsqueeze(now_Q, dim=0),dim=0)
        X = torch.matmul(now_Q,y)  # initialization
        for n in range(output_layers):
            step = self.steps[n]
            denoiser = self.denoisers[n]
            deblocker = self.deblocks[n]

            z = self.block1(X, y, now_mask, step)

            noise = denoiser(X)

            # X = z-self.block2()

            matrix_temp = torch.unsqueeze(torch.unsqueeze(
                step * torch.matmul(torch.transpose(now_mask, 0, 1), now_mask) - torch.eye(33 * 33).float().cuda(),
                dim=0), dim=0)
            # matrix_temp = (matrix_temp,dim=1)
            # matrix_temp = matrix_temp.expand(-1,y.shape[1],-1,-1)

            X = z - torch.matmul(matrix_temp, noise)

            X = self.together(X,S,H,L)
            X = X - deblocker(X)

            X = torch.cat(torch.split(X, split_size_or_sections=33, dim=2), dim=1)
            X = torch.cat(torch.split(X, split_size_or_sections=33, dim=3), dim=1)
            # inputs = torch.transpose(torch.reshape(inputs, [-1, 33*33]),0,1)
            X = torch.reshape(X, [S, H * L, 33 * 33])
            X = torch.unsqueeze(X, dim=3)

        X = self.together(X, S, H, L)
        return X


    def sampling(self,A,inputs):
        # inputs = torch.squeeze(inputs)
        # inputs = torch.reshape(inputs,[-1,33*33])  # 矩阵向量hua
        # inputs = torch.squeeze(inputs,dim=1)
        H = int(inputs.shape[2] / 33)
        L = int(inputs.shape[3] / 33)
        S = inputs.shape[0]
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=33, dim=2), dim=1)
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=33, dim=3), dim=1)
        # inputs = torch.transpose(torch.reshape(inputs, [-1, 33*33]),0,1)
        inputs = torch.reshape(inputs, [S, H * L, 33 * 33])
        inputs = torch.unsqueeze(inputs, dim=3)
        A_temp = torch.unsqueeze(torch.unsqueeze(A, dim=0),dim=0)
        # A_temp = A_temp.expand([-1, inputs.shape[1], -1, -1])
        outputs = torch.matmul(A_temp, inputs)
        # inputs = torch.reshape(inputs, [-1, 33 * 33])
        # inputs = torch.unsqueeze(inputs,dim=2)
        # outputs = torch.matmul(sampling_matrix, inputs)
        return outputs

    def block1(self,X,y,A,step):
        # X = torch.squeeze(X)
        # X = torch.transpose(torch.reshape(X, [-1, 33 * 33]),0,1)  # 矩阵向量hua
        A = torch.unsqueeze(torch.unsqueeze(A,dim=0),dim=0)
        # A = A.expand(-1, y.shape[1], -1, -1)
        outputs = torch.matmul(torch.transpose(A, 2, 3), y-torch.matmul(A, X))
        outputs = step * outputs + X
        # outputs = torch.unsqueeze(torch.reshape(torch.transpose(outputs,0,1),[-1,33,33]),dim=1)
        return outputs

    def together(self,inputs,S,H,L):
        inputs = torch.squeeze(inputs,dim=3)
        inputs = torch.reshape(inputs, [S, H * L, 33, 33])
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=H, dim=1), dim=3)
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=1, dim=1), dim=2)
        return inputs


class AMP_net_Deblock(Module):
    def __init__(self,layer_num, A):
        super().__init__()
        self.layer_num = layer_num
        self.denoisers = []
        self.deblocks = []
        self.steps = []
        self.register_parameter("A",nn.Parameter(torch.from_numpy(A).float(),requires_grad=True))
        self.register_parameter("Q", nn.Parameter(torch.from_numpy(np.transpose(A)).float(), requires_grad=True))
        for n in range(layer_num):
            self.denoisers.append(Denoiser())
            self.deblocks.append(Deblocker())
            self.register_parameter("step_" + str(n + 1), nn.Parameter(torch.tensor(1.0),requires_grad=True))
            self.steps.append(eval("self.step_" + str(n + 1)))
        for n,denoiser in enumerate(self.denoisers):
            self.add_module("denoiser_"+str(n+1),denoiser)
        for n,deblock in enumerate(self.deblocks):
            self.add_module("deblock_"+str(n+1),deblock)

    def forward(self, inputs, sampling_matrix_mask, output_layers):
        """
        此处就是前向传播，返回每一层的输出
        :param inputs: 此处的inputs 为图像数据
        :return:
        """
        H = int(inputs.shape[2]/33)
        L = int(inputs.shape[3]/33)
        S = inputs.shape[0]

        now_mask = self.A*sampling_matrix_mask
        now_Q = torch.transpose(sampling_matrix_mask,1,2)*self.Q
        y = self.sampling(now_mask, inputs)  # sampling

        now_Q = torch.unsqueeze(now_Q, dim=1)
        X = torch.matmul(now_Q,y)  # initialization
        for n in range(output_layers):
            step = self.steps[n]
            denoiser = self.denoisers[n]
            deblocker = self.deblocks[n]

            z = self.block1(X, y, now_mask, step)

            noise = denoiser(X)

            # X = z-self.block2()

            matrix_temp = torch.unsqueeze(step * torch.matmul(torch.transpose(now_mask,1,2), now_mask) - torch.eye(33 * 33).float().cuda(),dim=1)
            # matrix_temp = (matrix_temp,dim=1)
            # matrix_temp = matrix_temp.expand(-1,y.shape[1],-1,-1)

            X = z - torch.matmul(matrix_temp, noise)

            X = self.together(X,S,H,L)
            X = X - deblocker(X)

            X = torch.cat(torch.split(X, split_size_or_sections=33, dim=2), dim=1)
            X = torch.cat(torch.split(X, split_size_or_sections=33, dim=3), dim=1)
            # inputs = torch.transpose(torch.reshape(inputs, [-1, 33*33]),0,1)
            X = torch.reshape(X, [S, H * L, 33 * 33])
            X = torch.unsqueeze(X, dim=3)

        X = self.together(X, S, H, L)
        return X


    def sampling(self,A,inputs):
        # inputs = torch.squeeze(inputs)
        # inputs = torch.reshape(inputs,[-1,33*33])  # 矩阵向量hua
        # inputs = torch.squeeze(inputs,dim=1)
        H = int(inputs.shape[2] / 33)
        L = int(inputs.shape[3] / 33)
        S = inputs.shape[0]
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=33, dim=2), dim=1)
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=33, dim=3), dim=1)
        # inputs = torch.transpose(torch.reshape(inputs, [-1, 33*33]),0,1)
        inputs = torch.reshape(inputs, [S, H * L, 33 * 33])
        inputs = torch.unsqueeze(inputs, dim=3)
        A_temp = torch.unsqueeze(A, dim=1)
        # A_temp = A_temp.expand([-1, inputs.shape[1], -1, -1])
        outputs = torch.matmul(A_temp, inputs)
        # inputs = torch.reshape(inputs, [-1, 33 * 33])
        # inputs = torch.unsqueeze(inputs,dim=2)
        # outputs = torch.matmul(sampling_matrix, inputs)
        return outputs

    def block1(self,X,y,A,step):
        # X = torch.squeeze(X)
        # X = torch.transpose(torch.reshape(X, [-1, 33 * 33]),0,1)  # 矩阵向量hua
        A = torch.unsqueeze(A,dim=1)
        # A = A.expand(-1, y.shape[1], -1, -1)
        outputs = torch.matmul(torch.transpose(A, 2, 3), y-torch.matmul(A, X))
        outputs = step * outputs + X
        # outputs = torch.unsqueeze(torch.reshape(torch.transpose(outputs,0,1),[-1,33,33]),dim=1)
        return outputs

    def together(self,inputs,S,H,L):
        inputs = torch.squeeze(inputs,dim=3)
        inputs = torch.reshape(inputs, [S, H * L, 33, 33])
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=H, dim=1), dim=3)
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=1, dim=1), dim=2)
        return inputs


def compute_loss(outputs, target):
    loss = []
    for output in outputs:
        loss.append(torch.mean((output - target) ** 2))
    return loss


def get_final_loss(loss_all):
    output = 0
    for loss in loss_all:
        output += loss
    return output


def get_loss(outputs, noise_all, Xs, H, target, sigma=0.01):
    loss1 = torch.mean((outputs[-1] - target) ** 2)
    loss2 = torch.mean(torch.abs(outputs[-1] - target))
    num = 0
    for n in range(len(noise_all)):
        num += 1
        X = Xs[n]
        noise = noise_all[n]
        loss2 += torch.mean((noise - torch.matmul(H, target - X)) ** 2)

    return loss1, loss2

def train(model, opt, train_loader, epoch, batch_size, CS_ratio,PhaseNum):
    model.train()
    n = 0
    for data,_ in train_loader:
        n = n + 1
        opt.zero_grad()  # 清空梯度
        data = torch.unsqueeze(data,dim=1)
        data = Variable(data.float().cuda())
        outputs= model(data,PhaseNum)

        # loss_all = compute_loss(outputs,data)
        # loss = get_final_loss(loss_all)
        # loss = torch.mean((outputs[-1]-target)**2)

        loss = torch.mean((outputs-data)**2)
        loss.backward()
        opt.step()
        if n % 25 == 0:
            output = "CS_ratio: %d [%02d/%02d] loss: %.4f " % (
            CS_ratio, epoch, batch_size * n, loss.data.item())
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


def get_reconstruction_mask(test=50):
    data = torch.zeros([math.ceil(1089*0.5),1089])
    random_num = math.ceil(1089*(test/100))
    data[0:random_num,:] = 1
    return data


def get_val_result(model,PhaseNum,CS_ratio, is_cuda=True):
    model.eval()
    with torch.no_grad():
        test_set_path = "../../dataset/Set11"
        # test_set_path = "../../dataset/BSR_bsds500/BSR/BSDS500/data/images/test"
        test_set_path = glob.glob(test_set_path + '/*.tif')
        ImgNum = len(test_set_path)  # 测试图像的数量
        PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
        SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)
        PSNR_CS_ratios = np.zeros([1, 50], dtype=np.float32)
        SSIM_CS_ratios = np.zeros([1, 50], dtype=np.float32)
        model.eval()
        sub_path = "../../results_c4/AMP_net_deblock/"+CS_ratio

        n = 0
        CS_ratio_temp = [25]
        for CS_ratio in CS_ratio_temp:
            save_path = sub_path + "/images_generated"
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            save_path = os.path.join(save_path, str(CS_ratio))
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            print(n+1)
            for img_no in range(ImgNum):
                imgName = test_set_path[img_no]  # 当前图像的名字

                [Iorg, row, col, Ipad, row_new, col_new] = imread_CS_py(imgName)
                Icol = img2col_py(Ipad, 33) / 255.0  # 返回 行向量化后的图像数据
                Ipad /= 255.0
                # Img_input = np.dot(Icol, Phi_input)  # 压缩感知降采样
                # Img_output = Icol

                if is_cuda:
                    inputs = Variable(torch.from_numpy(Ipad.astype('float32')).cuda())
                else:
                    inputs = Variable(torch.from_numpy(Ipad.astype('float32')))
                # if model.network == "ista_plus" or model.network == "ista":
                #     output, _ = model(inputs)
                # else:
                #     output = model(inputs)
                inputs = torch.unsqueeze(torch.unsqueeze(inputs,dim=0),dim=0)
                sampling_matrix_mask = get_mask(1, CS_ratio)

                # sampling_matrix_mask = get_reconstruction_mask(CS_ratio)

                if is_cuda:
                    sampling_matrix_mask = Variable(sampling_matrix_mask.float().cuda())
                else:
                    sampling_matrix_mask = Variable(sampling_matrix_mask.float())

                outputs = model(inputs,sampling_matrix_mask, PhaseNum)
                outputs = torch.squeeze(outputs)
                if is_cuda:
                    outputs = outputs.cpu().data.numpy()
                else:
                    outputs = outputs.data.numpy()

                images_recovered = outputs[0:row, 0:col] * 255

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

            PSNR_CS_ratios[0,n] = np.mean(PSNR_All)
            SSIM_CS_ratios[0, n] = np.mean(SSIM_All)
            output_file = open(sub_path + "/Set11_PSNR_1_to_50.txt", 'a')
            output_file.write("%.4f " % (np.mean(PSNR_All)))
            output_file.close()
            output_file = open(sub_path + "/Set11_SSIM_1_to_50.txt", 'a')
            output_file.write("%.4f " % (np.mean(SSIM_All)))
            output_file.close()
            n+=1
    return PSNR_CS_ratios,SSIM_CS_ratios


def load_sampling_matrix(CS_ratio):
    path = "../../dataset/sampling_matrix"
    data = io.loadmat(os.path.join(path, str(CS_ratio) + '.mat'))['sampling_matrix']
    return data


def get_Q(data_set,A):
    A = torch.from_numpy(A)
    n = 0
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=len(data_set),
                                shuffle=True, num_workers=2)
    for data, target in data_loader:
        data = torch.transpose(torch.reshape(data, [-1, 33 * 33]), 0, 1)
        target = torch.transpose(torch.reshape(target, [-1, 33 * 33]), 0, 1)
        y = torch.matmul(A.float(),data.float())
        x = target.float()
        if n==0:
            ys = y
            Xs = x
            n = 1
        else:
            ys = torch.cat([ys,y],dim=1)
            Xs = torch.cat([Xs,x],dim=1)
    Q = torch.matmul(torch.matmul(Xs,torch.transpose(ys,0,1)),
                     torch.inverse(torch.matmul(ys, torch.transpose(ys, 0, 1))))
    return Q.numpy()


if __name__ == "__main__":
    model_name = "AMP_Net_deblock"
    # model_name = "AMP_Net_deblock_RA"

    CS_ratios = [0.01,0.1,0.25,0.4]
    max_CS_ratio = 50
    phase = 6
    CS_ratio = 50


    path = os.path.join("../../results_c4", model_name, str(CS_ratio),str(phase), "best_model.pkl")

    A = load_sampling_matrix(max_CS_ratio)

    model = AMP_net_Deblock(phase,A)
    # model.cuda()
    model.load_state_dict(torch.load(path))
    print("Start")
    psnrs, ssims = get_val_result(model, phase, str(CS_ratio), is_cuda=True)  # test AMP_net

    print_str = " %.4f %.4f %.4f %.4f %.4f %.4f %.4f" % (
     psnrs[0, 0], psnrs[0, 1], psnrs[0, 2],
    psnrs[0, 3], psnrs[0, 4], psnrs[0, 5], psnrs[0, 6])
    print(print_str)

    print_str = " %.4f %.4f %.4f %.4f %.4f %.4f %.4f" % (
        ssims[0, 0], ssims[0, 1], ssims[0, 2],
        ssims[0, 3], ssims[0, 4], ssims[0, 5], ssims[0, 6])
    print(print_str)
