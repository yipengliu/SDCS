from torch import nn
from dataset import dataset
import torch
from torch.autograd import Variable
import os
import numpy as np
import glob
from utils import *
from scipy import io


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


def get_mask(data_batch,test=0):
    data = torch.zeros([data_batch,math.ceil(1089*0.5),1089])
    for n in range(data_batch):
        if test==0:
            random_num = math.ceil(1089*(random.randint(1,50)/100))
        else:
            random_num = math.ceil(1089*(test/100))
        data[n,0:random_num,:] = 1
    return data


def get_loss(outputs, target,):
    loss = torch.mean((outputs - target) ** 2)
    return loss


def train(model, opt, train_loader, epoch, batch_size, PhaseNum, H,CS_ratio):
    model.train()
    n = 0
    for data, target in train_loader:
        n = n + 1
        opt.zero_grad()  # 清空梯度
        data, target = torch.transpose(torch.reshape(data, [-1, 33 * 33]), 0, 1), torch.transpose(
            torch.reshape(target, [-1, 33 * 33]), 0, 1)
        data, target = Variable(data.float().cuda()), Variable(target.float().cuda())

        data_batch = data.shape[1]
        sampling_matrix_mask = get_mask(data_batch)
        sampling_matrix_mask = Variable(sampling_matrix_mask.float().cuda())
        outputs= model(data, sampling_matrix_mask)

        # loss_all = compute_loss(outputs,data)
        # loss = get_final_loss(loss_all)
        # loss = torch.mean((outputs[-1]-target)**2)
        loss= get_loss(outputs, target)
        loss.backward()
        opt.step()
        if n % 25 == 0:
            output = "CS_ratio: %d PhaseNum: %d [%02d/%02d] loss: %.4f " % (
            CS_ratio, PhaseNum, epoch, batch_size * n, loss.data.item())
            # output = "[%02d/%02d] cost: %.4f, cost_sym: %.4f \n" % (epoch, batch_size*n,
            #                                        cost.data.item(),cost_sym.data.item())
            print(output)


def get_val_result(model, is_cuda=True):
    val_CS_ratios = [50, 40, 30, 25, 10, 4, 1]
    test_set_path = "../../dataset/BSR_bsds500/BSR/BSDS500/data/images/val"
    test_set_path = glob.glob(test_set_path + '/*.tif')
    ImgNum = len(test_set_path)  # 测试图像的数量
    PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
    PSNR_CS_ratios = np.zeros([1, len(val_CS_ratios)], dtype=np.float32)
    model.eval()
    n=0
    with torch.no_grad():
        for CS_ratio in val_CS_ratios:
            for img_no in range(ImgNum):
                imgName = test_set_path[img_no]  # 当前图像的名字

                [Iorg, row, col, Ipad, row_new, col_new] = imread_CS_py(imgName)
                Icol = img2col_py(Ipad, 33) / 255.0  # 返回 行向量化后的图像数据
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
                sampling_matrix_mask = get_mask(inputs.shape[1], CS_ratio)
                sampling_matrix_mask = Variable(sampling_matrix_mask.float().cuda())
                output = model(inputs,sampling_matrix_mask)
                if is_cuda:
                    output = output.cpu().data.numpy()
                else:
                    output = output.data.numpy()
                images_recovered = col2im_CS_py(output, row, col, row_new, col_new)
                rec_PSNR = psnr(images_recovered * 255, Iorg)  # 计算PSNR的值
                PSNR_All[0, img_no] = rec_PSNR
            PSNR_CS_ratios[0, n] = np.mean(PSNR_All)
            n+=1

    return PSNR_CS_ratios


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



if __name__ == "__main__":
    is_cuda = True
    CS_ratio = 50  # 4, 10, 25, 30, 40, 50
    # n_output = 1089
    # PhaseNumbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # block 数目为 5
    learning_rate = 0.0001
    EpochNum = 100
    batch_size = 64
    net_name = "DPDNN"
    results_saving_path = "../../results_c4"

    if not os.path.exists(results_saving_path):
        os.mkdir(results_saving_path)

    results_saving_path = os.path.join(results_saving_path, net_name)
    if not os.path.exists(results_saving_path):
        os.mkdir(results_saving_path)

    print('Load Data...')  # jiazaishuju

    train_dataset = dataset(root='../../dataset',train=True, transform=None,
                            target_transform=None)
    print(len(train_dataset))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=2)
    A = load_sampling_matrix(CS_ratio)
    # Q = get_Q(train_dataset, A)
    H = torch.from_numpy(np.matmul(np.transpose(A), A) - np.eye(33 * 33)).float().cuda()
    model = DPDNN(A, 6)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.cuda()

    sub_path = os.path.join(results_saving_path, str(CS_ratio))

    if not os.path.exists(sub_path):
        os.mkdir(sub_path)

    best_psnr = 0
    for epoch in range(1, EpochNum + 1):
        if epoch == 101:
            opt.defaults['lr'] *= 0.2
        psnr_cs_ratios = get_val_result(model)
        train(model, opt, train_loader, epoch, batch_size, 6, H,CS_ratio)
        psnr_cs_ratios = get_val_result(model)
        mean_psnr = np.mean(psnr_cs_ratios)

        print_str = "epoch: %d  psnr: mean %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f" % (
         epoch, mean_psnr, psnr_cs_ratios[0, 0], psnr_cs_ratios[0, 1], psnr_cs_ratios[0, 2],
        psnr_cs_ratios[0, 3], psnr_cs_ratios[0, 4], psnr_cs_ratios[0, 5], psnr_cs_ratios[0, 6])
        print(print_str)

        output_file = open(sub_path + "/log_PSNR.txt", 'a')
        output_file.write("PSNR: %.4f\n" % (mean_psnr))
        output_file.close()

        if mean_psnr > best_psnr:
            best_psnr = mean_psnr
            output_file = open(sub_path + "/log_PSNR_best.txt", 'a')
            output_file.write("PSNR: %.4f\n" % (best_psnr))
            output_file.close()
            torch.save(model.state_dict(), sub_path + "/best_model.pkl")
