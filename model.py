import torch
import torch.nn as nn
import torch.nn.functional as F
# import numpy as np

class ConcatFusion(nn.Module):
    def __init__(self, input_dim=6336, output_dim=12):
        super(ConcatFusion, self).__init__()

        self.fc_out = nn.Linear(input_dim, output_dim)

    def forward(self, x, y):
        # output = self.fc_x(x) + self.fc_y(y)
        fused_feature = torch.cat((x, y), dim=1)
        output=self.fc_out(fused_feature)
        return  x,y,output

class SumFusion(nn.Module):
    def __init__(self, input_dim=3168, output_dim=12):
        super(SumFusion, self).__init__()
        self.fc_x = nn.Linear(input_dim, output_dim)
        self.fc_y = nn.Linear(input_dim, output_dim)
        # self.fc_out = nn.Linear(input_dim, output_dim)

    def forward(self, x, y):
        # output = self.fc_x(x) + self.fc_y(y)
        # fused_feature = x + y
        output = self.fc_x(x) + self.fc_y(y)

        return  x, y,output

class FiLM(nn.Module):
    """
    FiLM: Visual Reasoning with a General Conditioning Layer,
    https://arxiv.org/pdf/1709.07871.pdf.
    revised for mid-concat case
    """

    def __init__(self, input_dim=3168, dim=3168, output_dim=12):
        super(FiLM, self).__init__()

        self.dim = input_dim
        self.fc_x = nn.Linear(input_dim, 2 * dim)
        self.fc_y = nn.Linear(input_dim, 2 * dim)
        self.fc_out = nn.Linear(2*dim, output_dim)

        # self.x_film = x_film

    def forward(self, x, y):

        gamma_x, beta_x = torch.split(self.fc_y(x), self.dim, 1)
        gamma_y, beta_y = torch.split(self.fc_y(y), self.dim, 1)

        x_new = gamma_y * x + beta_y
        y_new = gamma_x * y + beta_x

        output = torch.cat((x_new, y_new), dim=1)
        output = self.fc_out(output)

        return x_new, y_new, output


class GatedFusion(nn.Module):
    """
    Efficient Large-Scale Multi-Modal Classification,
    https://arxiv.org/pdf/1802.02892.pdf.
    revised for mid-concat case
    """

    def __init__(self, input_dim=3168, dim=3168, output_dim=12):
        super(GatedFusion, self).__init__()

        # self.fc_x = nn.Linear(input_dim, dim)
        # self.fc_y = nn.Linear(input_dim, dim)
        self.fc_out = nn.Linear(2 * dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        # out_x = self.fc_x(x)
        # out_y = self.fc_y(y)
        out_x = x
        out_y = y

        gate_x = self.sigmoid(out_x)
        y_new = torch.mul(gate_x, out_y)

        gate_y = self.sigmoid(out_y)
        x_new = torch.mul(gate_y, out_x)

        output = torch.cat((x_new, y_new), dim=1)
        output = self.fc_out(output)

        return x_new, y_new, output


class cnn_layers_1(nn.Module):
    """
    CNN layers applied on acc sensor data to generate pre-softmax
    ---
    params for __init__():
        input_size: e.g. 1
        num_classes: e.g. 6
    forward():
        Input: data
        Output: pre-softmax
    """
    def __init__(self, input_size):
        super().__init__()

        # Extract features, 2D conv layers
        self.features = nn.Sequential(
            nn.Conv2d(input_size, 64, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Conv2d(64, 64, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Conv2d(64, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Conv2d(32, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 16, -1)#[bsz, 16, 1, 198]

        return x


class cnn_layers_2(nn.Module):
    """
    CNN layers applied on acc sensor data to generate pre-softmax
    ---
    params for __init__():
        input_size: e.g. 1
        num_classes: e.g. 6
    forward():
        Input: data
        Output: pre-softmax
    """
    def __init__(self, input_size):
        super().__init__()

        # Extract features, 2D conv layers
        self.features = nn.Sequential(
            nn.Conv2d(input_size, 64, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Conv2d(64, 64, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Conv2d(64, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Conv2d(32, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 16, -1)

        return x


class Encoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.acc_cnn_layers = cnn_layers_1(input_size)
        self.gyr_cnn_layers = cnn_layers_2(input_size)

    def forward(self, x1, x2):

        acc_output = self.acc_cnn_layers(x1)
        gyro_output = self.gyr_cnn_layers(x2)

        return acc_output, gyro_output



class MyMMModel(nn.Module):
    """Model for human-activity-recognition."""
    def __init__(self, opt,input_size, num_classes):
        super().__init__()
        self.opt=opt
        self.encoder = Encoder(input_size)


        self.im = nn.Linear(3168, 1584)
        self.drop_im = nn.Dropout(p = 0.5)
        self.mu1 = nn.Parameter(torch.ones(1, 16,1))
        self.mu2 = nn.Parameter(torch.ones(1, 16,1))

        if self.opt.fusion=='Sum':
            self.fusion =SumFusion()
        elif self.opt.fusion == 'FiLM':
            self.fusion =FiLM()
        elif self.opt.fusion == 'Gated':
            self.fusion =GatedFusion()
        elif self.opt.fusion == 'Concat':
            self.fusion =ConcatFusion()
        # self.reg = nn.Linear(shape_list[-1], 1)






    def forward(self, x1, x2):
        # m1,m2,output = None,None,None
        acc_output, gyro_output = self.encoder(x1, x2)

        a=acc_output
        g=gyro_output
        # fused_feature, _ = self.gru(fused_feature)
        if self.opt.eluercos:
            ra, pa = self.mu1 * torch.cos(a), self.mu1 * torch.sin(a)
            # print(ra)
            rg, pg = self.mu2 * torch.cos(g), self.mu2 * torch.sin(g)

            ra=ra.reshape(ra.shape[0], -1)
            rg=rg.reshape(rg.shape[0], -1)
            pa=pa.reshape(pa.shape[0], -1)
            pg=pg.reshape(pg.shape[0], -1)
            ra, pa = self.drop_im(ra), self.drop_im(pa)
            rg, pg = self.drop_im(rg), self.drop_im(pg)
            ra, pa = self.im(ra), self.im(pa)
            rg, pg = self.im(rg), self.im(pg)

            ra, pa = torch.relu(ra), torch.relu(pa)
            rg, pg = torch.relu(rg), torch.relu(pg)
            ra_theta = torch.atan2(pa, ra)  # 计算角度
            rg_theta = torch.atan2(pg, rg)
            # rea, ima = self.reg(ra), self.reg(ra)
            # reg, img = self.reg(rg), self.reg(rg)
            # fea_a=ra+pa
            # fea_g=rg+pg
            fea_a = torch.cat((ra, pa), dim=1)
            fea_g = torch.cat((rg, pg), dim=1)
        else:
            fea_a, fea_g=a,g
            fea_a, fea_g = fea_a.contiguous().view(-1, 3168),fea_g.contiguous().view(-1, 3168)
            ra_theta, rg_theta=None,None
        if self.opt.fusion=='Concat':
        # fused_feature =torch.cat((acc_output, gyro_output), dim=1)
            m1,m2,output = self.fusion(fea_a, fea_g)
        elif self.opt.fusion=='Sum':
            m1,m2,output =self.fusion(fea_a, fea_g)
        elif self.opt.fusion == 'FiLM':
            m1,m2,output =self.fusion(fea_a, fea_g)
        elif self.opt.fusion == 'Gated':
            m1,m2,output =self.fusion(fea_a, fea_g)
            # fused_feature = fused_feature.contiguous().view(fused_feature.size(0), 6336)
            # a = fea_a.contiguous().view(-1, 3168)
            # g = fea_g.contiguous().view(-1, 3168)


        return output,m1,m2,ra_theta,rg_theta


class MySingleModel(nn.Module):
    """Model for human-activity-recognition."""
    def __init__(self, input_size, num_classes, modality):
        super().__init__()

        if modality == 'acc':
            self.encoder = cnn_layers_1(input_size)
        else:
            self.encoder = cnn_layers_2(input_size)


        self.gru = nn.GRU(198, 120, 2, batch_first=True)

        # Classify output, fully connected layers
        self.classifier = nn.Sequential(

            nn.Linear(1920, 1280),
            nn.BatchNorm1d(1280),
            nn.ReLU(inplace=True),

            nn.Linear(1280, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Linear(128, num_classes),
            )

    def forward(self, x):

        x = self.encoder(x)

        x = x.view(x.size(0), 16, -1)
        x, _ = self.gru(x)

        x = x.contiguous().view(x.size(0), 1920)
        output = self.classifier(x)


        return output





