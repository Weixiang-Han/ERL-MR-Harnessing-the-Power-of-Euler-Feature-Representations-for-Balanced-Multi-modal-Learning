from __future__ import print_function

import os
import sys
import argparse
import time
import math
import numpy as np
import torch.nn as nn

# import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer, save_model
import torch.nn.functional as F
from model import MyMMModel,MySingleModel
from model_mhad import MyMMModel_mhad

import data_pre as data
import data_pre_mhad as datam

from communication import COMM
from solver import MinNormSolver_
from solver import gradient_normalizers

from torch.autograd import Variable
try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--use', type=bool, default=False,
                        help='n')
    parser.add_argument('--fusion', type=str, default='Concat',
                        help='n')
    parser.add_argument('--eluercos', type=bool, default=False,
                        help='mmcosine')
    parser.add_argument('--pareto', type=bool, default=False,
                        help='mmcosine')
    parser.add_argument('--ours', type=bool, default=False,
                        help='mmcosine')

    parser.add_argument('--mmcosine', type=bool, default=False,
                        help='mmcosine')
    parser.add_argument('--scaling', default=10, type=float, help='scaling parameter in mmCosine')

    parser.add_argument('--proto', default=False, type=bool, help='scaling parameter in mmCosine')
    parser.add_argument('--momentum_coef', default=0.2, type=float, help='scaling parameter in mmCosine')
    parser.add_argument('--alpha_p',default=0.5, type=float, help='alpha in OGM-GE')


    parser.add_argument('--OGM', default=False, type=bool, help='scaling parameter in mmCosine')
    parser.add_argument('--modulation', default='OGM_GE', type=str, help='scaling parameter in mmCosine')
    parser.add_argument('--modulation_starts', default=0, type=float, help='scaling parameter in mmCosine')
    parser.add_argument('--modulation_ends', default=200, type=float, help='scaling parameter in mmCosine')
    parser.add_argument('--alpha',default=0.5, type=float, help='alpha in OGM-GE')


    # FL
    parser.add_argument('--usr_id', type=int, default=100,
                        help='user id')
    parser.add_argument('--fl_epoch', type=int, default=5,
                        help='communication to server after the epoch of local training')
    parser.add_argument('--server_address', type=str, default="192.168.83.1",
                        help='server_address')
    parser.add_argument('--local_modality', type=str, default='both',
                        help='indicator of local modality')  # both, acc, gyr

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=20,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=399,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='50,100,150',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.9,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='MyMMmodel')
    parser.add_argument('--approach', type=str, default='mmFL')
    parser.add_argument('--dataset', type=str, default='USC-HAR',
                        choices=['USC-HAR', 'UTD-MHAD', 'ours'], help='dataset')
    parser.add_argument('--num_class', type=int, default=11,
                        help='num_class,usc12,nhad11')
    parser.add_argument('--num_of_train', type=int, default=100,
                        help='num_of_train')
    parser.add_argument('--num_of_test', type=int, default=250,
                        help='num_of_test')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=int, default='1',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.result_path = './'

    if not os.path.isdir(opt.result_path):
        os.makedirs(opt.result_path)

    return opt


# construct data loader
def set_loader(opt):
    # load data (already normalized)
    if opt.usc:
        data_path = "D:/FLcode/Harmony-main/harmony-USC/USC-data/node_{}/".format(opt.usr_id)
    else:
        data_path = "D:/FLcode/Harmony-main/harmony-MHAD/client/MHAD-data/node_{}/".format(opt.usr_id)

    if opt.usc:

    # load data (already normalized)
        x_train_1 = np.load(data_path + "x_train_1.npy")
        x_train_2 = np.load(data_path + "x_train_2.npy")
        y_train = np.load(data_path + "y_train.npy")
        x_test_1 = np.load(data_path + "x_test_1.npy")
        x_test_2 = np.load(data_path + "x_test_2.npy")
        y_test = np.load(data_path + "y_test.npy")
    else:
        x_train_1 = np.load(data_path + "x1_train.npy")
        x_train_2 = np.load(data_path + "x2_train.npy")
        y_train = np.load(data_path + "y_train.npy")
        x_test_1 = np.load(data_path + "x1_test.npy")
        x_test_2 = np.load(data_path + "x2_test.npy")
        y_test = np.load(data_path + "y_test.npy")

    print(x_train_1.shape)
    print(x_train_2.shape)
    print(x_test_1.shape)
    print(x_test_2.shape)
    # x_train_1=np.array([x_train_1[0:20] for i in range(5)]).reshape(-1,3,200)
    # print(x_train_1.shape)
    print(y_train.shape)


    if opt.usc:
        # train_dataset = data.Multimodal_imdataset(reduced_x_train_1, x_train_2, y_train, reduced_labels)
        train_dataset = data.Multimodal_dataset( x_train_1, x_train_2, y_train)

        test_dataset = data.Multimodal_dataset(x_test_1, x_test_2, y_test)
    else:
        train_dataset = datam.Multimodal_dataset(x_train_1, x_train_2, y_train)
        test_dataset = datam.Multimodal_dataset(x_test_1, x_test_2, y_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size,
        num_workers=opt.num_workers, pin_memory=True, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.batch_size,
        num_workers=opt.num_workers, pin_memory=True, shuffle=True)

    return train_loader, test_loader


def set_model(opt):
    if opt.usc:
        model = MyMMModel(opt,input_size=1, num_classes=opt.num_class)
    else:
        model = MyMMModel_mhad(opt,input_size=1, num_classes=opt.num_class)


    criterion = torch.nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion

def EU_dist(x1, x2):
    d_matrix = torch.zeros(x1.shape[0], x2.shape[0]).to(x1.device)
    for i in range(x1.shape[0]):
        for j in range(x2.shape[0]):
            d = torch.sqrt(torch.dot((x1[i] - x2[j]), (x1[i] - x2[j])))
            d_matrix[i, j] = d
    return d_matrix


def calculate_prototype(args, model, dataloader, epoch, a_proto=None, v_proto=None):
    # if args.dataset == 'VGGSound':
    #     n_classes = 309
    # elif args.dataset == 'KineticSound':
    #     n_classes = 31
    # elif args.dataset == 'CREMAD':
    #     n_classes = 6
    # elif args.dataset == 'AVE':
    #     n_classes = 28
    # elif args.dataset == 'CGMNIST':
    #     n_classes = 10
    # else:
    #     raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))
    n_classes=12
    a_prototypes = torch.zeros(n_classes, 3168).cuda()
    g_prototypes = torch.zeros(n_classes, 3168).cuda()
    count_class = [0 for _ in range(n_classes)]

    # calculate prototype
    model.eval()
    with torch.no_grad():
        sample_count = 0
        all_num = len(dataloader)
        for idx, (input_data1, input_data2, labels) in enumerate(dataloader):
            if torch.cuda.is_available():
                input_data1 = input_data1.cuda()
                input_data2 = input_data2.cuda()
                labels = labels.cuda()

            # warm-up learning rate
            # warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)
            # compute loss
            output, a, g = model(input_data1, input_data2)


            for c, l in enumerate(labels):
                l = l.long()
                count_class[l] += 1
                a_prototypes[l, :] += a[c, :]
                g_prototypes[l, :] += g[c, :]
                # if l == 22:
                #     print('fea_a', a[c, :], audio_prototypes[l, :])

            sample_count += 1
            if args.dataset == 'AVE':
                pass
            else:
                if sample_count >= all_num // 10:
                    break
    for c in range(a_prototypes.shape[0]):
        a_prototypes[c, :] /= count_class[c]
        g_prototypes[c, :] /= count_class[c]

    if epoch <= 0:
        a_prototypes = a_prototypes
        g_prototypes = g_prototypes
    else:
        a_prototypes = (1 - args.momentum_coef) * a_prototypes + args.momentum_coef * a_proto
        g_prototypes = (1 - args.momentum_coef) * g_prototypes + args.momentum_coef * v_proto
    return a_prototypes, g_prototypes


def train_single(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (input_data1, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            input_data1 = input_data1.cuda()
            labels = labels.cuda()
        bsz = input_data1.shape[0]

        # warm-up learning rate
        # warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        output = model(input_data1)
        loss = criterion(output, labels)

        acc, _ = accuracy(output, labels, topk=(1, 5))

        # update metric
        losses.update(loss.item(), bsz)
        top1.update(acc[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg


def train_multi(train_loader, model, criterion, optimizer, epoch, opt,a_proto=None,g_proto=None):
    """one epoch training"""

    model.train()
    softmax = nn.Softmax(dim=1)
    relu = nn.ReLU(inplace=True)
    tanh = nn.Tanh()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    topa=AverageMeter()
    topg=AverageMeter()
    at=[]
    gt=[]
    theta_a=None
    theta_g=None

    end = time.time()
    for idx, (input_data1, input_data2, labels) in enumerate(train_loader):
        loss_data = {}
        grads = {}
        scale = {}
        data_time.update(time.time() - end)
        if torch.cuda.is_available():
            input_data1 = input_data1.cuda()
            input_data2 = input_data2.cuda()
            labels = labels.cuda()
        bsz = input_data1.shape[0]

        # warm-up learning rate
        # warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)
        # compute loss
        output,a,g,ra_theta,rg_theta = model(input_data1, input_data2)

        # print(model.fusion.fc_x.weight.shape)
        # loss_c=1 - torch.cos(torch.mean(ra_theta) - torch.mean(rg_theta))
        # print(loss_c.shape)
        # print(model.classifier1.weight.shape)
        if opt.mmcosine:

            if opt.fusion=='Sum':
                out_a = torch.mm(F.normalize(a,dim=1), F.normalize(torch.transpose(model.fusion.fc_x.weight, 0, 1),dim=0))   # w[n_classes,feature_dim*2]->W[feature_dim, n_classes], norm at dim 0.
                out_g = torch.mm(F.normalize(g,dim=1), F.normalize(torch.transpose(model.fusion.fc_y.weight, 0, 1),dim=0))
            else:
                out_a = torch.mm(F.normalize(a, dim=1),
                                 F.normalize(torch.transpose(model.fusion.fc_out.weight, 0, 1)[:3168, :],
                                             dim=0))  # w[n_classes,feature_dim*2]->W[feature_dim, n_classes], norm at dim 0.
                out_g = torch.mm(F.normalize(g, dim=1),
                                 F.normalize(torch.transpose(model.fusion.fc_out.weight, 0, 1)[3168:, :], dim=0))
            out_a=out_a*opt.scaling
            out_g=out_g*opt.scaling
            output = out_a + out_g
        else:
        # 65.14-65.28  65 59-61.x
        # 62.71        61.85  57-60.71
            if opt.fusion=='Sum':
                out_a = (torch.mm(a, torch.transpose(model.fusion.fc_x.weight, 0, 1)) +
                     model.fusion.fc_x.bias / 2)
                out_g = (torch.mm(g, torch.transpose(model.fusion.fc_y.weight, 0, 1)) +
                 model.fusion.fc_y.bias / 2)
            else:

                out_a = (torch.mm(a, torch.transpose(model.fusion.fc_out.weight, 0, 1)[:3168,:]) +
                         model.fusion.fc_out.bias / 2)
                out_g = (torch.mm(g, torch.transpose(model.fusion.fc_out.weight, 0, 1)[3168:, :]) +
                         model.fusion.fc_out.bias / 2)
        if not opt.proto:

            optimizer.zero_grad()
            if opt.pareto and opt.ours:
                output, a, g, ra_theta, rg_theta = model(input_data1, input_data2)

                loss_o = criterion(output, labels)
                loss_data[0] = loss_o.item()
                loss_o.backward(retain_graph=True)
                grads[0] = []
                for param in model.parameters():
                    if param.grad is not None:
                        # print(param)
                        # print('-----')
                        grads[0].append(Variable(param.grad.data.clone(), requires_grad=False))
                grads[0]=grads[0][0:35]
                #
                #
                #
                # # for param, saved_grad in zip(params, saved_grads):
                # #     param.grad = saved_grad
                #
                optimizer.zero_grad()
                # loss_c=1 - torch.cos(torch.mean(ra_theta) - torch.mean(rg_theta))
                output, a, g, ra_theta, rg_theta = model(input_data1, input_data2)
                loss_c=2*(1 - torch.mean(torch.cosine_similarity(ra_theta, rg_theta, dim=0)))
                loss_data[1] = loss_c.item()
                loss_c.backward(retain_graph=True)
                grads[1] = []
                for param in model.parameters():
                    if param.grad is not None:

                        grads[1].append(Variable(param.grad.data.clone(), requires_grad=False))
                # # print(len(grads[0]))
                # # print(len(grads[1]))


                gn = gradient_normalizers(grads, loss_data, 'loss+')
                for t in range(2):
                    for gr_i in range(len(grads[t])):
                        grads[t][gr_i] = grads[t][gr_i] / gn[t]
                # print(grads)
                sol, min_norm = MinNormSolver_.find_min_norm_element([grads[t] for t in range(2)])
                for i, t in enumerate(range(2)):
                    scale[t] = float(sol[i])
                optimizer.zero_grad()
                # #
                # output, a, g, ra_theta, rg_theta = model(input_data1, input_data2)
                loss_o = criterion(output, labels)
                # loss_c = 1 - torch.cos(torch.mean(ra_theta) - torch.mean(rg_theta))
                loss_c =1-torch.mean(torch.cosine_similarity(ra_theta, rg_theta, dim=0))+criterion(out_a, labels)+criterion(out_g, labels)
                print(scale[0],scale[1])
                loss = scale[0]*loss_o+scale[1]*loss_c
            else:
                loss = criterion(output, labels)
        else:
            a_sim = -EU_dist(a, a_proto)  # B x n_class
            g_sim = -EU_dist(g, g_proto)
            # B x n_class

            if opt.modulation == 'Proto' and opt.modulation_starts <= epoch <= opt.modulation_ends:

                score_a_p = sum([softmax(a_sim)[i][labels[i]] for i in range(a_sim.size(0))])
                score_g_p = sum([softmax(g_sim)[i][labels[i]] for i in range(g_sim.size(0))])
                ratio_a_p = score_a_p / score_g_p

                score_g = sum([softmax(out_g)[i][labels[i]] for i in range(out_g.size(0))])
                score_a = sum([softmax(out_a)[i][labels[i]] for i in range(out_a.size(0))])
                ratio_a = score_a / score_g

                loss_proto_a = criterion(a_sim, labels)
                loss_proto_v = criterion(g_sim, labels)

                if ratio_a_p > 1:
                    beta = 0  # audio coef
                    lam = 1 * opt.alpha_p  # visual coef
                elif ratio_a_p < 1:
                    beta = 1 * opt.alpha_p
                    lam = 0
                else:
                    beta = 0
                    lam = 0
                loss = criterion(output, labels) + beta * loss_proto_a + lam * loss_proto_v
                loss_v = criterion(out_g, labels)
                loss_a = criterion(out_a, labels)
            else:
                loss = criterion(output, labels)
                loss_proto_v = criterion(g_sim, labels)
                loss_proto_a = criterion(a_sim, labels)
                loss_v = criterion(out_g, labels)
                loss_a = criterion(out_a, labels)

                score_a_p = sum([softmax(a_sim)[i][labels[i]] for i in range(a_sim.size(0))])
                score_g_p = sum([softmax(g_sim)[i][labels[i]] for i in range(g_sim.size(0))])
                ratio_a_p = score_a_p / score_g_p
                score_g = sum([softmax(out_g)[i][labels[i]] for i in range(out_g.size(0))])
                score_a = sum([softmax(out_a)[i][labels[i]] for i in range(out_a.size(0))])
                ratio_a = score_a / score_g

        acc, _ = accuracy(output, labels, topk=(1, 5))
        acca, _ = accuracy(out_a, labels, topk=(1, 5))
        accg, _ = accuracy(out_g, labels, topk=(1, 5))


        # update metric
        losses.update(loss.item(), bsz)
        top1.update(acc[0], bsz)
        topa.update(acca[0], bsz)
        topg.update(accg[0], bsz)

        # optimizer.zero_grad()
        loss.backward()
        # loss.backward(retain_graph=True)

        if opt.OGM:
            score_g = sum([softmax(out_g)[i][labels[i]] for i in range(out_g.size(0))])
            score_a = sum([softmax(out_a)[i][labels[i]] for i in range(out_a.size(0))])

            ratio_g = score_g / score_a
            ratio_a = 1 / ratio_g

            """
            Below is the Eq.(10) in our CVPR paper:
                    1 - tanh(alpha * rho_t_u), if rho_t_u > 1
            k_t_u =
                    1,                         else
            coeff_u is k_t_u, where t means iteration steps and u is modality indicator, either a or v.
            """

            if ratio_g > 1:
                coeff_g = 1 - tanh(opt.alpha * relu(ratio_g))
                coeff_a = 1
            else:
                coeff_a = 1 - tanh(opt.alpha * relu(ratio_a))
                coeff_g = 1

            # if args.use_tensorboard:
            #     iteration = epoch * len(dataloader) + step
            #     writer.add_scalar('data/ratio v', ratio_v, iteration)
            #     writer.add_scalar('data/coefficient v', coeff_v, iteration)
            #     writer.add_scalar('data/coefficient a', coeff_a, iteration)

            if opt.modulation_starts <= epoch <= opt.modulation_ends: # bug fixed
                for name, parms in model.named_parameters():
                    # print(name)
                    # layer = str(name).split('.')[1]
                    layer = str(name)

                    if 'acc' in layer :
                        if opt.modulation == 'OGM_GE':
                            # bug fixed
                            parms.grad = parms.grad * coeff_a + \
                                         torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                        elif opt.modulation == 'OGM':
                            parms.grad *= coeff_a

                    if 'gyr' in layer :
                        if opt.modulation == 'OGM_GE':
                            # bug fixed
                            parms.grad = parms.grad * coeff_g + \
                                         torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                        elif opt.modulation == 'OGM':
                            parms.grad *= coeff_g
            else:
                pass

        # SGD

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Acca@1 {topa.val:.3f} ({topa.avg:.3f})\t'
                    'Accg@1 {topg.val:.3f} ({topg.avg:.3f})\t'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1,topa=topa,topg=topg))
            sys.stdout.flush()

    return losses.avg,at,gt


def validate_single(val_loader, model, criterion, opt):
    """validation"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    topa=AverageMeter()
    topg=AverageMeter()

    confusion = np.zeros((opt.num_class, opt.num_class))

    with torch.no_grad():
        end = time.time()
        for idx, (input_data1, labels) in enumerate(val_loader):

            if torch.cuda.is_available():
                input_data1 = input_data1.float().cuda()
                labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output,a,g = model(input_data1)
            loss = criterion(output, labels)
            out_a = (torch.mm(a, torch.transpose(model.classifier1.weight[:, 3168:], 0, 1)) +
                     model.classifier1.bias / 2)
            out_g = (torch.mm(g, torch.transpose(model.classifier1.weight[:, :3168], 0, 1)) +
                     model.classifier1.bias / 2)
            softmax = torch.nn.Softmax(dim=1)
            # prediction = softmax(out)
            pred_a = softmax(out_a)
            pred_g = softmax(out_g)
            # update metric
            acc, _ = accuracy(output, labels, topk=(1, 5))
            acca, _ = accuracy(pred_a, labels, topk=(1, 5))
            accg, _ = accuracy(pred_g, labels, topk=(1, 5))
            losses.update(loss.item(), bsz)
            top1.update(acc[0], bsz)
            topa.update(acca[0], bsz)
            topg.update(accg[0], bsz)

            # calculate and store confusion matrix
            rows = labels.cpu().numpy()
            cols = output.max(1)[1].cpu().numpy()
            for label_index in range(labels.shape[0]):
                confusion[rows[label_index], cols[label_index]] += 1

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acca@1 {topa.val:.3f} ({topa.avg:.3f})\t'
                      'Accg@1 {topg.val:.3f} ({topg.avg:.3f})\t'
                .format(
                    idx, len(val_loader), batch_time=batch_time,
                    loss=losses, top1=top1,topa=topa,topg=topg))

    return losses.avg, top1.avg, confusion


def validate_multi(val_loader, model, criterion, opt):
    """validation"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    topa=AverageMeter()
    topg=AverageMeter()
    lossesa=AverageMeter()
    lossesg=AverageMeter()

    confusion = np.zeros((opt.num_class, opt.num_class))

    with torch.no_grad():
        end = time.time()
        for idx, (input_data1, input_data2, labels) in enumerate(val_loader):

            if torch.cuda.is_available():
                input_data1 = input_data1.float().cuda()
                input_data2 = input_data2.float().cuda()
                labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output,a,g,ra_theta,rg_theta = model(input_data1, input_data2)
            if opt.mmcosine:
                if opt.fusion == 'Sum':
                    out_a = torch.mm(F.normalize(a, dim=1),
                                     F.normalize(torch.transpose(model.fusion.fc_x.weight, 0, 1),
                                                 dim=0))  # w[n_classes,feature_dim*2]->W[feature_dim, n_classes], norm at dim 0.
                    out_g = torch.mm(F.normalize(g, dim=1),
                                     F.normalize(torch.transpose(model.fusion.fc_y.weight, 0, 1), dim=0))
                else:
                    out_a = torch.mm(F.normalize(a, dim=1),
                                     F.normalize(torch.transpose(model.fusion.fc_out.weight, 0, 1)[:3168, :],
                                                 dim=0))  # w[n_classes,feature_dim*2]->W[feature_dim, n_classes], norm at dim 0.
                    out_g = torch.mm(F.normalize(g, dim=1),
                                     F.normalize(torch.transpose(model.fusion.fc_out.weight, 0, 1)[3168:, :], dim=0))
                out_a = out_a * opt.scaling
                out_g = out_g * opt.scaling
                output=out_a+out_g

            else:
                if opt.fusion == 'Sum':
                    out_a = (torch.mm(a, torch.transpose(model.fusion.fc_x.weight, 0, 1)) +
                             model.fusion.fc_x.bias / 2)
                    out_g = (torch.mm(g, torch.transpose(model.fusion.fc_y.weight, 0, 1)) +
                             model.fusion.fc_y.bias / 2)
                else:

                    out_a = (torch.mm(a, torch.transpose(model.fusion.fc_out.weight, 0, 1)[:3168, :]) +
                             model.fusion.fc_out.bias / 2)
                    out_g = (torch.mm(g, torch.transpose(model.fusion.fc_out.weight, 0, 1)[3168:, :]) +
                             model.fusion.fc_out.bias / 2)
            loss = criterion(output, labels)
            lossa = criterion(out_a, labels)
            lossg = criterion(out_g, labels)

            # update metric
            acc, _ = accuracy(output, labels, topk=(1, 5))
            acca, _ = accuracy(out_a, labels, topk=(1, 5))
            accg, _ = accuracy(out_g, labels, topk=(1, 5))
            losses.update(loss.item(), bsz)
            lossesa.update(lossa.item(), bsz)
            lossesg.update(lossg.item(), bsz)
            top1.update(acc[0], bsz)
            topa.update(acca[0], bsz)
            topg.update(accg[0], bsz)

            # calculate and store confusion matrix
            rows = labels.cpu().numpy()
            cols = output.max(1)[1].cpu().numpy()
            for label_index in range(labels.shape[0]):
                confusion[rows[label_index], cols[label_index]] += 1

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Lossa {lossesa.val:.4f} ({lossesa.avg:.4f})\t'
                      'Lossg {lossesg.val:.4f} ({lossesg.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t' 
                      'Acca@1 {topa.val:.3f} ({topa.avg:.3f})\t'
                      'Accg@1 {topg.val:.3f} ({topg.avg:.3f})\t'
                .format(
                    idx, len(val_loader), batch_time=batch_time,
                    loss=losses,lossesa=lossesa,lossesg=lossesg,top1=top1,topa=topa,topg=topg))

    return losses.avg, top1.avg, confusion,topa.avg,topg.avg


def get_model_array(model):
    params = []
    for param in model.parameters():
        if torch.cuda.is_available():
            params.extend(param.view(-1).cpu().detach().numpy())
        else:
            params.extend(param.view(-1).detach().numpy())
        # print(param)

    # model_params = params.cpu().numpy()
    model_params = np.array(params)
    print("Shape of model weight: ", model_params.shape)  # 39456

    return model_params


def reset_model_parameter(new_params, model):
    temp_index = 0

    with torch.no_grad():
        for param in model.parameters():

            # print(param.shape)

            if len(param.shape) == 2:

                para_len = int(param.shape[0] * param.shape[1])
                temp_weight = new_params[temp_index: temp_index + para_len].astype(float)
                param.copy_(torch.from_numpy(temp_weight.reshape(param.shape[0], param.shape[1])))
                temp_index += para_len

            elif len(param.shape) == 4:

                para_len = int(param.shape[0] * param.shape[1] * param.shape[2] * param.shape[3])
                temp_weight = new_params[temp_index: temp_index + para_len].astype(float)
                param.copy_(torch.from_numpy(
                    temp_weight.reshape(param.shape[0], param.shape[1], param.shape[2], param.shape[3])))
                temp_index += para_len

            else:
                para_len = param.shape[0]
                temp_weight = new_params[temp_index: temp_index + para_len].astype(float)
                param.copy_(torch.from_numpy(temp_weight))
                temp_index += para_len


def set_commu(opt):
    # prepare the communication module
    server_addr = opt.server_address

    user_id = opt.usr_id

    if opt.local_modality == "gyr":
        server_port = 9997
        # if opt.usr_id > 4:
        #     user_id = opt.usr_id - 5
    else:
        server_port = 9998

    comm = COMM(server_addr, server_port, user_id)

    comm.send2server('hello', -1)

    print(comm.recvfserver())

    comm.send2server(opt.local_modality, 1)

    return comm


def main():
    opt = parse_option()
    torch.manual_seed(42)

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)
    print(model)
    w_parameter_init = get_model_array(model)
    # build optimizer
    optimizer = set_optimizer(opt, model)

    best_acc = 0
    best_confusion = np.zeros((opt.num_class, opt.num_class))
    record_loss = np.zeros(opt.epochs)
    record_lossa = np.zeros(opt.epochs)
    record_lossg = np.zeros(opt.epochs)
    record_acc = np.zeros(opt.epochs)
    record_acca = np.zeros(opt.epochs)
    record_accg = np.zeros(opt.epochs)

    atheta = []
    gtheta = []
    # set up communication with sevrer
    # comm = set_commu(opt)
    epoch=0
    if opt.proto:
        a_proto, g_proto = calculate_prototype(opt, model, train_loader, epoch)
    # training routine
    for epoch in range(1, opt.epochs + 1):

        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        if opt.proto:
            loss = train_multi(train_loader, model, criterion, optimizer, epoch, opt,a_proto,g_proto)
        else:
            loss,at,gt = train_multi(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        atheta.extend(at)
        gtheta.extend(gt)
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        record_loss[epoch - 1] = loss
        if opt.proto:
            a_proto, g_proto = calculate_prototype(opt, model, train_loader, epoch, a_proto,
                                                        g_proto)

        # evaluation
        if opt.local_modality == 'both':
            loss, val_acc, confusion, acca, accg = validate_multi(val_loader, model, criterion, opt)
        else:
            loss, val_acc, confusion = validate_single(val_loader, model, criterion, opt)
        record_acc[epoch - 1] = val_acc
        record_acca[epoch - 1] = acca
        record_accg[epoch - 1] = accg


        if val_acc > best_acc:
            best_acc = val_acc
            best_confusion = confusion

        # # communication with the server every fl_epoch
        # if (epoch % opt.fl_epoch) == 0:
        #     ## send model update to the server
        #     print("Node {} sends weight to the server:".format(opt.usr_id))
        #     w_parameter = get_model_array(model)  # obtain the model parameters or gradients
        #     w_update = w_parameter - w_parameter_init
        #     comm.send2server(w_parameter, 0)
        #     # print('send shape:',len(w_update))
        #
        #     ## recieve aggregated model update from the server
        #     new_w_update, sig_stop = comm.recvOUF()
        #     print("Received weight from the server:", new_w_update)
        #     print("Received signal from the server:", sig_stop)
        #
        #     ## update the model according to the received weights
        #     # new_w = w_parameter_init + new_w_update
        #     new_w = new_w_update
        #
        #     reset_model_parameter(new_w, model)
        #     w_parameter_init = new_w


    # torch.save(atheta,'./atheta.txt')
    # torch.save(gtheta,'./gtheta.txt')

    # evaluation
    # np.savetxt(opt.result_path + "athetac.txt".format(opt.usr_id), atheta)
    # np.savetxt(opt.result_path + "gthetac.txt".format(opt.usr_id), gtheta)

    if opt.local_modality == 'both':
        loss, val_acc, confusion,acca,accg = validate_multi(val_loader, model, criterion, opt)
    else:
        loss, val_acc, confusion = validate_single(val_loader, model, criterion, opt)

    print("Testing accuracy of node {} is : multi-acc{},multi1-acc{},multi2-acc{}".format(opt.usr_id, val_acc,acca,accg))
    np.savetxt(opt.result_path + "record_loss_FiLM_org.txt".format(opt.usr_id), record_loss)
    np.savetxt(opt.result_path + "record_acc_FiLM_org.txt".format(opt.usr_id), record_acc)
    np.savetxt(opt.result_path + "record_acca_FiLM_org.txt".format(opt.usr_id), record_acca)
    np.savetxt(opt.result_path + "record_accg_FiLM_org.txt".format(opt.usr_id), record_accg)

    # np.savetxt(opt.result_path + "record_confusion.txt".format(opt.usr_id), confusion)

    # print("Save FL model!")
    # fl_model_path = "./save_uni_FL/{}_models/".format(opt.dataset)
    # if not os.path.isdir(fl_model_path):
    #     os.makedirs(fl_model_path)
    # if opt.local_modality == 'acc':
    #     save_model(model.encoder, optimizer, opt, opt.epochs, os.path.join(fl_model_path, 'last_acc.pth'))
    # else:
    #     save_model(model.encoder, optimizer, opt, opt.epochs, os.path.join(fl_model_path, 'last_gyr.pth'))

    # comm.disconnect(1)


if __name__ == '__main__':

    main()
