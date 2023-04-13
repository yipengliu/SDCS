from dataset import dataset
import torch
from torch.autograd import Variable
from torch import nn
import os
import numpy as np
import glob
from utils import *
from scipy import io


class DPDNN_encoder(nn.Module):
    def __init__(self,num_in,num_out):
        super().__init__()
        self.f1 = nn.Conv2d(num_in, 64, 3, padding=1)
        self.f2 = nn.Conv2d(64, 64, 3, padding=1)
        self.f3 = nn.Conv2d(64, 64, 3, padding=1)
        self.f4 = nn.Conv2d(64, 64, 3, padding=1, stride=2)


    def forward(self, inputs):
        outputs = nn.ReLU()(self.f1(inputs))
        outputs = nn.ReLU()(self.f2(outputs))
        output1 = nn.ReLU()(self.f3(outputs))
        output2 = nn.ReLU()(self.f4(output1))
        return output1, output2

class DPDNN_encoder_end(nn.Module):
    def __init__(self,num_in,num_out):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(num_in, num_out, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, inputs):
        outputs = self.encoder(inputs)
        return outputs


class DPDNN_decoder(nn.Module):
    def __init__(self,num_in,num_out):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Conv2d(num_in, 64, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, num_out, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, inputs):
        outputs = self.decoder(inputs)
        return outputs


class DPDNN_(nn.Module):
    def __init__(self):
        super().__init__()
        self.f1 = DPDNN_encoder(1,64)
        self.f2 = DPDNN_encoder(64,64)
        self.f3 = DPDNN_encoder(64, 64)
        self.f4 = DPDNN_encoder(64, 64)
        self.f5 = DPDNN_encoder_end(64, 64)

        self.d1 = DPDNN_decoder(128, 64)
        self.d2 = DPDNN_decoder(128, 64)
        self.d3 = DPDNN_decoder(128, 64)
        self.d4 = DPDNN_decoder(128, 64)

        self.decov1 = nn.ConvTranspose2d(64, 64, 3, stride=2,padding=1)
        self.decov2 = nn.ConvTranspose2d(64, 64, 3, stride=2,padding=1)
        self.decov3 = nn.ConvTranspose2d(64, 64, 3, stride=2,padding=1)
        self.decov4 = nn.ConvTranspose2d(64, 64, 3, stride=2,padding=1)

        self.conv = nn.Conv2d(64, 1, 3, padding=1)
    def forward(self, inputs):
        inputs = torch.unsqueeze(torch.reshape(torch.transpose(inputs, 0, 1), [-1, 33, 33]), dim=1)
        y1, y1_temp = self.f1(inputs)
        # y2 = nn.Upsample(size=[16,16])(y1)
        y2, y2_temp = self.f2(y1_temp)
        # y3 = nn.Upsample(size=[8, 8])(y2)
        y3, y3_temp = self.f3(y2_temp)
        # y4 = nn.Upsample(size=[4, 4])(y3)
        y4, y4_temp = self.f4(y3_temp)
        # y5 = nn.Upsample(size=[2, 2])(y4)
        y5 = self.f5(y4_temp)
        z6 = self.decov1(y5)
        z6 = torch.cat([z6, y4], dim=1)
        z5 = self.d1(z6)
        z5 = self.decov2(z5)
        z4 = torch.cat([z5, y3], dim=1)
        z4 = self.d2(z4)
        z4 = self.decov3(z4)
        z3 = torch.cat([z4, y2], dim=1)
        z3 = self.d3(z3)
        z3 = self.decov4(z3)
        z2 = torch.cat([z3, y1], dim=1)
        z2 = self.d4(z2)
        z1 = self.conv(z2)
        outputs = inputs + z1
        outputs = torch.transpose(torch.reshape(outputs, [-1, 33 * 33]), 0, 1)
        return outputs

class DPDNN(nn.Module):
    def __init__(self, A, T):
        super().__init__()
        self.T = T
        self.register_parameter("A", nn.Parameter(torch.from_numpy(A).float(), requires_grad=True))
        self.register_parameter("Q", nn.Parameter(torch.transpose(torch.from_numpy(A).float(),0,1), requires_grad=True))
        self.denoiser = DPDNN_()
        self.sigmas = []
        self.taos = []
        for n in range(T):
            self.register_parameter("sigma_"+str(n + 1), nn.Parameter(torch.Tensor([0.1]).float(), requires_grad=True))
            self.sigmas.append(eval("self.sigma_"+str(n + 1)))
            self.register_parameter("tao_" + str(n + 1), nn.Parameter(torch.Tensor([0.9]).float(), requires_grad=True))
            self.taos.append(eval("self.tao_" + str(n + 1)))

    def forward(self, inputs, sampling_matrix_mask):
        # all the inputs are the same
        now_mask = self.A * sampling_matrix_mask
        now_Q = torch.transpose(sampling_matrix_mask, 1, 2) * self.Q
        y = self.sampling(inputs,now_mask)  # 采样，获得y
        # inputs = torch.transpose(inputs,0,1)
        X = torch.matmul(now_Q, y)

        for n in range(self.T):
            tao = self.taos[n]
            sigma = self.sigmas[n]

            temp = torch.squeeze(X)
            temp = torch.transpose(temp, 0, 1)
            temp = self.denoiser(temp)
            temp = torch.transpose(temp, 0, 1)
            temp = torch.unsqueeze(temp, dim=2)

            X = sigma*torch.matmul(torch.transpose(now_mask,1,2),y-torch.matmul(now_mask,X))+(1-sigma*tao)*X+sigma*tao*temp
        return torch.transpose(torch.squeeze(X),0,1)

    def sampling(self, inputs, A):
        # inputs = torch.squeeze(inputs)
        # inputs = torch.reshape(inputs,[-1,33*33])  # 矩阵向量hua
        inputs = torch.transpose(inputs, 0, 1)
        inputs = torch.unsqueeze(inputs, dim=2)
        outputs = torch.matmul(A, inputs)
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
        sub_path = "../../results_c4/DPDNN"

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

            PSNR_CS_ratios[0, CS_ratio - 1] = np.mean(PSNR_All)
            SSIM_CS_ratios[0, CS_ratio -1] = np.mean(SSIM_All)
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
    model_name = "DPDNN"

    CS_ratio = 50
    phase = 9

    path = os.path.join("../../results_c4", model_name, str(CS_ratio), "best_model.pkl")
    A = load_sampling_matrix(CS_ratio)
    model = DPDNN(A, 6)
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
