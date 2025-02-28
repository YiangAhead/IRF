import argparse
import os
import logging
import random
import sys
import time
from albumentations.pytorch import ToTensorV2
from albumentations.core.composition import Compose
import albumentations as A
import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import losses
from CMUNet import CMUNet
from two_stream_dataloader import Train_Dataset, Val_Dataset, TwoStreamBatchSampler
from metrics import test_single_volume_dual
import ramps
from patch import devide_into_patchs, compute_hard_ratio, generate_patch_data
import matplotlib.pyplot as plt
from UNet_res import MyModel
import math
from models.nets.deeplabv3_plus import DeepLab

def seed_torch(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str,
                    default='UNet_CMUNet_001_patch4', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='CMUNet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=10000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--initial-lr', default=0.0, type=float,
                    metavar='LR', help='initial learning rate when using linear rampup')
parser.add_argument('--lr-rampup', default=0, type=int, metavar='EPOCHS',
                    help='length of learning rate rampup in the beginning')
parser.add_argument('--num_classes', type=int, default=2,
                    help='output channel of network')
parser.add_argument('--seed', type=int, default=3049, help='random seed')
parser.add_argument('--labeled_bs', type=int, default=4,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=64,
                    help='labeled data')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency_rampup', type=float,
                    default=181.0, help='consistency_rampup')
parser.add_argument('--consistency-scale', default=1.0, type=float, metavar='WEIGHT',
                    help='use consistency loss with given weight (default: None)')
parser.add_argument('--stabilization-rampup', default=181.0, type=float, metavar='EPOCHS',
                    help='length of the stabilization loss ramp-up')
parser.add_argument('--stable-threshold', default=0.5, type=float, metavar='THRESHOLD',
                    help='threshold for stable sample')
parser.add_argument('--stable-threshold-teacher', default=0.5, type=float, metavar='THRESHOLD',
                    help='threshold for stable sample')
parser.add_argument('--stabilization-scale', default=1.0, type=float, metavar='WEIGHT',
                    help='use stabilization loss with given weight (default: None)')
parser.add_argument('--logit-distance-cost', default=0.05, type=float, metavar='WEIGHT',
                    help='let the student model have two outputs and use an MSE loss '
                         'the logits with the given weight (default: only have one output)')

args = parser.parse_args()
seed_torch(args.seed)


def weight_up(current, max_epoch):
    phase = 1.0 - current / max_epoch
    return float(np.exp(-0.5 * phase * phase))

def linear_rampup(current, rampup_length):
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length

def cosine_rampdown(current, rampdown_length):
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))

def zero_cosine_rampdown(current, epochs):
    return float(.5 * (1.0 + np.cos((current - 1) * np.pi / epochs)))


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1-float(iter)/max_iter)**power)


def adjust_lr_rate(optimizer, iter, total_batch, max_epoch):
    lr = lr_poly(args.base_lr, iter, max_epoch*total_batch, 0.9)
    # optimizer.param_groups[0]['lr'] = lr
    for param_groups in optimizer.param_groups:
        param_groups['lr'] = lr
    return lr


def train(args, snapshot_path, writer):
    # 分割网络的学习率
    base_lr = args.base_lr
    # 需要分割的类别数
    num_classes = args.num_classes
    batch_size = args.batch_size
    # 最大迭代次数
    max_iterations = args.max_iterations

    def create_model(model_name):
        # Network definition
        if model_name == 'U_Net_Dual':
            model = U_Net_Dual(img_ch=3, output_ch=num_classes)
        elif model_name == 'CMUNet':
            model = CMUNet(img_ch=3, output_ch=num_classes, l=7, k=7)
        elif model_name == 'Unet_Res':
            model = MyModel(num_classes=2,in_channels=3)
        elif model_name == 'deeplab_dual':
            model = DeepLab(backbone='mobilenet', pretrained=True,num_classes=2)
        else:
            assert False, "Model {} not available".format(model_name)
        return model

    stu1_model = create_model("CMUNet")  # student1 model
    stu2_model = create_model("CMUNet")  # student2 model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    stu1_model.to(device)
    stu2_model.to(device)

    train_transform = Compose([
        A.RandomRotate90(),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomGamma(gamma_limit=(30, 150), p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.3, p=0.2),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.2),
        A.CLAHE(clip_limit=5.0, tile_grid_size=(5, 5), p=0.2),
        A.ShiftScaleRotate(shift_limit=0.1, rotate_limit=30, scale_limit=0.3, p=0.2),
        A.OneOf([
                A.GaussNoise(),
                A.GaussianBlur(),
                A.MotionBlur(),
                ], p=0.5),              
        A.Resize(256, 256),
        ToTensorV2(),
        # A.ToTensor(),
        # Trans.Normalize(),
    ])

    val_transform = Compose([
        A.Resize(256, 256),
        ToTensorV2(),
        # A.ToTensor(),
        # Trans.Normalize(),
    ])

    total_number = len(data_train)
    # print('total_number:', total_number)
    labeled_number = args.labeled_num
    # print('labeled_number:', labeled_number)
    # 有标签图像的序号是（0，labeled_number-1），无标签图像的序号是（labeled_number，total_number）
    labeled_idxs = list(range(0, labeled_number))
    unlabeled_idxs = list(range(labeled_number, total_number))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs)

    # 这里是数据加载器，dataloader
    trainloader = DataLoader(data_train, batch_sampler=batch_sampler)

    val_batch_size = 1
    valloader = DataLoader(data_val, batch_size=val_batch_size, shuffle=False)
    testloader = DataLoader(data_test, batch_size=val_batch_size, shuffle=False)

    # 进入模型训练模式
    stu1_model.train()
    stu2_model.train()

    # 两个分割网络的优化器
    stu1_optimizer = optim.SGD(stu1_model.parameters(), lr=base_lr,
                               momentum=0.9, weight_decay=0.0001)
    stu2_optimizer = optim.SGD(stu2_model.parameters(), lr=base_lr,
                               momentum=0.9, weight_decay=0.0001)

    # 损失函数
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    residual_logit_criterion = losses.symmetric_mse_loss
    eucliden_distance = losses.softmax_mse_loss

    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
        stabilization_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
        stabilization_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    iter_num = 0
    val_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance1 = 0.0
    best_performance2 = 0.0
    
    iterator = tqdm(range(max_epoch))
    noise_r = 0.2

    for epoch_num in iterator:
        total_batch = math.ceil(len(data_train) / args.batch_size)
        for i_batch, sampled_batch in enumerate(trainloader):
            itr = total_batch * epoch_num + i_batch
            
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume1_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            out, out2 = stu1_model(volume1_batch)
            image = volume1_batch
            label = label_batch.squeeze(1)
            logit = out[:, 0, :, :]
            pred = torch.softmax(out[:], dim=1)
            patch_num = 2
            ex_ratio = 20
            '''
            添加entropy-driven patch augmentation
            '''

            noise = torch.clamp(torch.randn_like(volume1_batch) * 0.1, -noise_r, noise_r)
            volume2_batch = new_image + noise
            label_batch = new_label
            # stu1_out1,stu2_out2是两个模型的输出结果,stu1e_out2,stu2e_out1是两个模型对加噪声图像的输出结果
            stu1_out1 = stu1_model(new_image)
            stu1e_out2 = stu1_model(volume2_batch)
            stu2_out2 = stu2_model(new_image)
            # print(f'stu2_out2:{stu2_out2[0].shape}')
            stu2e_out1 = stu2_model(volume2_batch)

            assert len(stu1_out1) == 2
          
            stu1_seg_logit, stu1_cons_logit = stu1_out1
            stu2_seg_logit, stu2_cons_logit = stu2_out2
            stu1e_seg_logit, stu1e_cons_logit = stu1e_out2
            stu2e_seg_logit, stu2e_cons_logit = stu2e_out1
            # 计算两个模型的损失

            #模型1：stu1----真实标签Yl被直接用于监督有标注分割结果Pl
            # (B, C, H, W)   (B, H, W)
            stu1_loss_ce = ce_loss(stu1_seg_logit[:args.labeled_bs].squeeze(1),
                                   label_batch[:][:args.labeled_bs].squeeze(1).long())

            # (B, C, H, W)   (B, 1, H, W)
            # print('-----------------------------')
            # print(torch.softmax(stu1_seg_logit[:args.labeled_bs], dim=1).shape)
            # print(label_batch[:args.labeled_bs].shape)
            # time.sleep(10)
            stu1_loss_dice = dice_loss(torch.softmax(stu1_seg_logit[:args.labeled_bs], dim=1),
                                       label_batch[:args.labeled_bs].unsqueeze(1))
            stu1_seg_loss = 0.5 * (stu1_loss_ce + stu1_loss_dice)
            # stu1_seg_loss = stu1_loss_ce

            #模型2：stu2----真实标签Yl被直接用于监督有标注分割结果Pl
            stu2_loss_ce = ce_loss(stu2_seg_logit[:args.labeled_bs].squeeze(1),
                                   label_batch[:][:args.labeled_bs].squeeze(1).long())
            stu2_loss_dice = dice_loss(torch.softmax(stu2_seg_logit[:args.labeled_bs], dim=1),
                                       label_batch[:args.labeled_bs].unsqueeze(1))
            stu2_seg_loss = 0.5 * (stu2_loss_ce + stu2_loss_dice)
            # stu2_seg_loss = stu2_loss_ce
            # stu1_seg_loss,stu2_seg_loss是最终目标损失函数的第一部分
            stu1_loss = stu1_seg_loss
            stu2_loss = stu2_seg_loss

            '''
            2.计算一致性约束部分的损失函数con_Q
            '''

            # 针对无标签图像

            # 一致性损失，args.consistency_scale是一个超参数,默认值是1，args.consistenct_rampup是一个超参数，默认值是181
            # consistency loss
            # 伴随变量，consistency_criterion是一个损失函数，这里是均方误差损失函数
            stu1_seg = Variable(stu1_seg_logit.detach().data, requires_grad=False)
            # 对图像加入噪声，计算 无噪声图像输入监督学习的模型stu1模型的分割结果 和 加噪声图像的分割结果 的一致性损失
            # stu1_consistency_loss1 = 0.05 * consistency_criterion(stu1_seg_logit[:args.labeled_bs],
            #                                                       stu1_seg[:args.labeled_bs])
            # stu1_consistency_loss2 = 0.05 * consistency_criterion(stu1e_cons_logit[:args.labeled_bs],
            #                                                       stu1_seg[:args.labeled_bs])
            #修改
            stu1_consistency_loss2 = 0.05 * consistency_criterion(stu1e_cons_logit[:args.labeled_bs],
                                                        stu1_seg[:args.labeled_bs])
            
            # 计算一致性损失的均值
            # stu1_consistency_loss_1 = torch.mean(stu1_consistency_loss1)
            stu1_consistency_loss_2 = torch.mean(stu1_consistency_loss2)
            # stu1_consistency_loss_Q = stu1_consistency_loss_1 + stu1_consistency_loss_2
            stu1_consistency_loss_Q = stu1_consistency_loss_2

            # stu1_consistency_loss是最终目标损失函数的第二部分
            stu1_loss += stu1_consistency_loss_Q

            # 与上同理
            stu2_seg = Variable(stu2_seg_logit.detach().data, requires_grad=False)
            # stu2_consistency_loss1 = 0.05 * consistency_criterion(stu2_cons_logit[:args.labeled_bs],
            #                                                       stu2_seg[:args.labeled_bs])
            # stu2_consistency_loss2 = 0.05 * consistency_criterion(stu2e_cons_logit[:args.labeled_bs],
            #                                                       stu2_seg[:args.labeled_bs])
            stu2_consistency_loss2 = 0.05 * consistency_criterion(stu2e_seg_logit[:args.labeled_bs],
                                                      stu2_seg[:args.labeled_bs])
            # stu2_consistency_loss_1 = torch.mean(stu2_consistency_loss1)
            stu2_consistency_loss_2 = torch.mean(stu2_consistency_loss2)
            # stu2_consistency_loss_Q = stu2_consistency_loss_1 + stu2_consistency_loss_2
            stu2_consistency_loss_Q = stu2_consistency_loss_2
            stu2_loss += stu2_consistency_loss_Q


            consistency_weight = args.consistency_scale * weight_up(epoch_num, max_epoch)

            '''
             Mutual correction
            '''

            # 反向传播
            stu1_optimizer.zero_grad()
            stu1_loss.backward()
            stu1_optimizer.step()

            stu2_optimizer.zero_grad()
            stu2_loss.backward()
            stu2_optimizer.step()

            # 动态调整学习率
            # lr_stu1 = adjust_learning_rate(stu1_optimizer, epoch_num, i_batch, len(trainloader), max_epoch)
            # lr_stu2 = adjust_learning_rate(stu2_optimizer, epoch_num, i_batch, len(trainloader), max_epoch)
            lr_stu1 = adjust_lr_rate(stu1_optimizer, itr, total_batch,max_epoch)
            lr_stu2 = adjust_lr_rate(stu2_optimizer, itr, total_batch,max_epoch)

            # 将训练过程中的值写入tensorboard日志文件
            # writer = SummaryWriter('snapshot_path' + '/log')

            logging.info("{} iterations per epoch".format(len(trainloader)))
            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_stu1, iter_num)
            writer.add_scalar('info/stu1_loss', stu1_loss, iter_num)
            writer.add_scalar('info/stu2_loss', stu2_loss, iter_num)
            writer.add_scalar('info/stu1_loss_ce&dice', stu1_seg_loss, iter_num)
            writer.add_scalar('info/stu2_loss_ce&dice', stu2_seg_loss, iter_num)
            writer.add_scalar('info/stu1_stabilization_loss',
                              stu1_stabilization_loss, iter_num)
            writer.add_scalar('info/stu2_stabilization_loss',
                              stu2_stabilization_loss, iter_num)
            # writer.add_scalar('info/stu1_consistency_loss_Q',
            #                   stu1_consistency_loss_Q, iter_num)
            # writer.add_scalar('info/stu1_consistency_loss_P',
            #                   stu1_consistency_loss_P, iter_num)
            # writer.add_scalar('info/stu2_consistency_loss_Q',
            #                   stu2_consistency_loss_Q, iter_num)
            # writer.add_scalar('info/stu2_consistency_loss_P',
            #                   stu2_consistency_loss_P, iter_num)
            logging.info(
                'iteration %d : stu1_loss : %f, stu2_loss: %f' %
                (iter_num, stu1_loss.item(), stu2_loss.item()))

            # 每20次迭代，将训练过程中的图像写入tensorboard日志文件
            if iter_num % 100 == 0:
                image = new_image[1, :, :, :]
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(stu1_seg_logit, dim=1), dim=1, keepdim=True)
                # print('out', outputs)
                writer.add_image('train/Prediction',
                                 outputs[1, ...] * 50, iter_num)
                labs = new_label[1, ...].unsqueeze(0) * 50

                writer.add_image('train/GroundTruth', labs.squeeze(1), iter_num)

                # 可视化 logit（例如，选择第一个类别的 logit 值）
                logit_image = stu1_seg_logit[1, 0, ...]  # 选择第一个类别的 logit 值
                logit_image = logit_image.unsqueeze(0)  # 添加 batch 维度，使形状为 (1, height, width)

                # 归一化 logit 的范围到 [0, 1]，以便正确显示
                logit_image = (logit_image - logit_image.min()) / (logit_image.max() - logit_image.min())  # 归一化到 [0, 1]

                # 转换为 Numpy 数组，并应用 colormap（热图色图）
                logit_image_np = logit_image.squeeze().cpu().detach().numpy()  # 转换为 NumPy 数组
                logit_image_colormap = plt.cm.jet(logit_image_np)  # 使用 jet 色图
                logit_image_colormap = torch.tensor(logit_image_colormap).permute(2, 0, 1)  # 调整维度为 (C, H, W)

                writer.add_image('train/Logit', logit_image_colormap, iter_num)  # 可视化 logit 图像
            if iter_num > 0 and iter_num % 100 == 0:
                stu1_model.eval()
                metric_list_1 = 0.0
                metric_list_2 = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    val_num = i_batch + (iter_num / 100) - 1
                    metric_i, writer = test_single_volume_dual(
                        sampled_batch["image"], sampled_batch["label"], stu1_model, iter_num, writer, i_batch,
                        classes=num_classes)
                    metric_list_1 += np.array(metric_i)
                metric_list_1 = metric_list_1 / len(data_val)

                performance1 = np.mean(metric_list_1, axis=0)[0]
                mean_hd95 = np.mean(metric_list_1, axis=0)[1]
                writer.add_scalar('info/val_mean_dice_model1', performance1, iter_num)
                writer.add_scalar('info/val_mean_hd95_model1', mean_hd95, iter_num)

                if performance1 > best_performance1:
                    best_performance1 = performance1
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance1, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(stu1_model.state_dict(), save_mode_path)
                    torch.save(stu1_model.state_dict(), save_best)

                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance1, mean_hd95))
                
                for i_batch, sampled_batch in enumerate(testloader):
                    val_num = i_batch + (iter_num / 100) - 1
                    metric_i, writer = test_single_volume_dual(
                        sampled_batch["image"], sampled_batch["label"], stu1_model, iter_num, writer, i_batch,
                        classes=num_classes)
                    metric_list_2 += np.array(metric_i)
                metric_list_2 = metric_list_2 / len(data_test)
                # print('metric_list',metric_list)
                # for class_i in range(num_classes-1):
                #     writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                #                       metric_list[class_i, 0], iter_num)
                #     writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
                #                       metric_list[class_i, 1], iter_num)

                performance1_test  = np.mean(metric_list_2, axis=0)[0]
                mean_hd95 = np.mean(metric_list_2, axis=0)[1]
                writer.add_scalar('info/test_mean_dice_model1', performance1_test, iter_num)
                writer.add_scalar('info/test_mean_hd95_model1', mean_hd95, iter_num)

                stu1_model.train()
                
                stu2_model.eval()
                metric_list_1 = 0.0
                metric_list_2 = 0.0
                
                
                for i_batch, sampled_batch in enumerate(valloader):
                    val_num = i_batch + (iter_num / 100) - 1
                    metric_i, writer = test_single_volume_dual(
                        sampled_batch["image"], sampled_batch["label"], stu2_model, iter_num, writer, i_batch,
                        classes=num_classes)
                    metric_list_1 += np.array(metric_i)
                metric_list_1 = metric_list_1 / len(data_val)
                # print('metric_list',metric_list)
                # for class_i in range(num_classes-1):
                #     writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                #                       metric_list[class_i, 0], iter_num)
                #     writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
                #                       metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list_1, axis=0)[0]
                mean_hd95 = np.mean(metric_list_1, axis=0)[1]
                writer.add_scalar('info/val_mean_dice_model2', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95_model2', mean_hd95, iter_num)

                if performance > best_performance2:
                    best_performance2 = performance
                    save_mode2_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}_model2.pth'.format(
                                                      iter_num, round(best_performance2, 4)))
                    save_best2 = os.path.join(snapshot_path,
                                             '{}_best_model_model2.pth'.format(args.model))
                    torch.save(stu2_model.state_dict(), save_mode2_path)
                    torch.save(stu2_model.state_dict(), save_best2)

                logging.info(
                    'model2_iteration %d : model2_mean_dice : %f model2_mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                
                for i_batch, sampled_batch in enumerate(testloader):
                    val_num = i_batch + (iter_num / 100) - 1
                    metric_i, writer = test_single_volume_dual(
                        sampled_batch["image"], sampled_batch["label"], stu2_model, iter_num, writer, i_batch,
                        classes=num_classes)
                    metric_list_2 += np.array(metric_i)
                metric_list_2 = metric_list_2 / len(data_test)
                # print('metric_list',metric_list)
                # for class_i in range(num_classes-1):
                #     writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                #                       metric_list[class_i, 0], iter_num)
                #     writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
                #                       metric_list[class_i, 1], iter_num)

                performance_test = np.mean(metric_list_2, axis=0)[0]
                mean_hd95 = np.mean(metric_list_2, axis=0)[1]
                writer.add_scalar('info/test_mean_dice_model2', performance_test, iter_num)
                writer.add_scalar('info/test_mean_hd95_model2', mean_hd95, iter_num)
                stu2_model.train()
                

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(stu1_model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
                
                save_mode2_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '_model2.pth')
                torch.save(stu2_model.state_dict(), save_mode2_path)
                logging.info("save model to {}".format(save_mode2_path))

            if iter_num >= max_iterations:
                break

        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # 模型文件保存路径，可以自己设定自己的文件保存路径
    snapshot_path = "checkpoints/model/{}_{}_labeled/{}".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    # 生成日志文件
    log_path = snapshot_path + "/log.txt"
    if os.path.exists(log_path):
        os.remove(log_path)
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    writer = SummaryWriter(f'runs/{args.exp}')
    train(args, snapshot_path, writer)
