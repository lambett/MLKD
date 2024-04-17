import os
import argparse
import sys

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import time
import torch.optim as optim

import csv

# cudnn.benchmark = True

from mdistiller.models import cifar_model_dict, cifar10_model_dict, imagenet_model_dict
from mdistiller.distillers import distiller_dict
from mdistiller.dataset import get_dataset
from mdistiller.engine.utils import load_checkpoint, log_msg, AverageMeter, accuracy, validate_t
from mdistiller.engine.cfg import CFG as cfg
from mdistiller.engine.cfg import show_cfg
from mdistiller.engine import trainer_dict
from mdistiller.models.cifar.wrapper_sp import wrapper_sp, wrapper_sp_s
from mdistiller.engine.losses import CompLoss, SupConLoss, HCR, DisLoss

# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def save_train_teacher(model_t, optimizer_t, model_path, best_acc):
    state = {
        'epoch': 60,
        'model': model_t.state_dict(),
        't_acc': best_acc,
        'optimizer': optimizer_t.state_dict(),
    }
    print('saving the teacher model!')
    torch.save(state, model_path)
    return


def main(cfg, resume, opts):
    experiment_name = cfg.EXPERIMENT.NAME
    if experiment_name == "":
        experiment_name = cfg.EXPERIMENT.TAG
    tags = cfg.EXPERIMENT.TAG.split(",")
    if opts:
        addtional_tags = ["{}:{}".format(k, v) for k, v in zip(opts[::2], opts[1::2])]
        tags += addtional_tags
        experiment_name += ",".join(addtional_tags)
    experiment_name = os.path.join(cfg.EXPERIMENT.PROJECT, experiment_name)
    if cfg.LOG.WANDB:
        try:
            import wandb

            wandb.init(project=cfg.EXPERIMENT.PROJECT, name=experiment_name, tags=tags)
        except:
            print(log_msg("Failed to use WANDB", "INFO"))
            cfg.LOG.WANDB = False

    # 数据集的不同比例
    # cfg.few_ratio = 1

    # cfg & loggers
    show_cfg(cfg)
    # init dataloader & models
    train_loader_weak, train_loader_strong, val_loader, num_data, num_classes = get_dataset(cfg)
    aux_loader = get_dataset(cfg, eval=True)
    n_cls = num_classes

    # distillation
    print(log_msg("Loading teacher model", "INFO"))
    if cfg.DATASET.TYPE == "cifar10":
        net, pretrain_model_path = cifar10_model_dict[cfg.DISTILLER.TEACHER]
    else:
        net, pretrain_model_path = cifar_model_dict[cfg.DISTILLER.TEACHER]
    print(pretrain_model_path)
    assert (
            pretrain_model_path is not None
    ), "no pretrain model for teacher {}".format(cfg.DISTILLER.TEACHER)
    model_teacher = net(num_classes=num_classes)
    try:
        model_teacher.load_state_dict(load_checkpoint(pretrain_model_path)["model"])
    except:
        model_teacher.load_state_dict(load_checkpoint(pretrain_model_path)["state_dict"])
    model_student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](
        num_classes=num_classes
    )

    # 添加投影头
    model_teacher = wrapper_sp(model_teacher, 128).cuda()
    model_student = wrapper_sp_s(model_student, 128).cuda()

    if cfg.DATASET.TYPE == "cifar10":
        model_t_path = f'/root/code/MLKD/save_packd_t/cifar10/{net.__name__}_embed0.pth'
    else:
        model_t_path = f'/root/code/MLKD/save_packd_t/{net.__name__}_embed0.pth'

    print(model_t_path)
    if os.path.exists(model_t_path):
        model_teacher.load_state_dict(torch.load(
            model_t_path, map_location='cpu')['model'])
        model_teacher.eval()
    else:
        criterion_comp = CompLoss(n_cls).cuda()
        criterion_dis = DisLoss(n_cls, 128, model_teacher, aux_loader, temperature=0.1).cuda()

        t_optimizer = optim.SGD([{'params': model_teacher.backbone.parameters(), 'lr': 0.0},
                                 {'params': model_teacher.proj_head.parameters(),
                                  'lr': 0.01},
                                 {'params': model_teacher.classifier.parameters(), 'lr': 0.05}],
                                momentum=0.9,
                                weight_decay=5e-4)
        model_teacher.eval()
        # train ssp_head
        for epoch in range(100):
            model_teacher.eval()
            loss_comp_record = AverageMeter()
            loss_dis_record = AverageMeter()
            loss_record = AverageMeter()
            acc_record = AverageMeter()

            start = time.time()
            for idx, data in enumerate(train_loader_weak):
                x, target, _ = data
                x = x.cuda()
                target = target.cuda()

                feat, out, proj_x, proj_logit = model_teacher(x, bb_grad=False)
                batch = int(x.size(0))

                feature = proj_x.detach()
                feature = F.normalize(feature, dim=1)
                dis_loss = criterion_dis(feature, target) * 0.05
                comp_loss = criterion_comp(feature, criterion_dis.prototypes, target)
                # loss = F.cross_entropy(proj_logit, target)
                loss = dis_loss + comp_loss + F.cross_entropy(proj_logit, target)
                loss.requires_grad_(True)
                t_optimizer.zero_grad()
                loss.backward()
                t_optimizer.step()
                batch_acc = accuracy(proj_logit, target, topk=(1,))[0]
                loss_dis_record.update(dis_loss.item(), batch)
                loss_comp_record.update(comp_loss.item(), batch)
                loss_record.update(loss.item(), batch)
                acc_record.update(batch_acc.item(), batch)
            run_time = time.time() - start
            info = f'teacher_train_Epoch:{epoch}/{100}\t run_time:{run_time:.3f}\t t_loss:{loss_record.avg:.3f}\t comp_loss:{loss_comp_record.avg:.3f}\t dis_loss:{loss_dis_record.avg:.3f}\t t_acc:{acc_record.avg:.2f}\t'
            print(info, flush=True)
        save_train_teacher(model_teacher, t_optimizer, model_t_path, 99)

    # validate teacher accuracy
    teacher_acc, _ = validate_t(val_loader, model_teacher)
    print('teacher accuracy: ', teacher_acc, flush=True)

    if cfg.DATASET.TYPE == "cifar10":
        N = 60000
    else:
        N = 50000
    s_confi = torch.zeros(N).cuda()
    w_confi = torch.zeros(N).cuda()
    w_probs = torch.zeros(N, n_cls).cuda()
    s_probs = torch.zeros(N, n_cls).cuda()
    labels = torch.ones(N).long().cuda()

    print("Loading labels .....")

    idx = 0
    for data, data_f in zip(train_loader_weak, train_loader_strong):
        input, target, index = data
        input = input.cuda()
        target = target.cuda()
        labels[index] = target
        input_f, target_f, index_f = data_f
        input_f = input_f.cuda()

        _, logit, _, _ = model_teacher(input)
        _, logit_f, _, _ = model_teacher(input_f)

        w_prob = F.softmax(logit, dim=1)
        S_prob = F.softmax(logit_f, dim=1)
        w_probs[index] = w_prob
        s_probs[index] = S_prob
        idx += 1
        print("\r", end='')
        print("idx:{}/782".format(idx), "#" * (idx // 11), end="")
        sys.stdout.flush()

    print("\nThe label are loaded")
    # labels = labels.cuda()

    w_confi = w_confi + 0.98
    s_confi = s_confi + 0.00005

    w_mask = w_probs[labels >= 0, labels] > w_confi[labels >= 0]
    s_mask = s_probs[labels >= 0, labels] > s_confi[labels >= 0]

    # y = list(range(0, 50000))
    # w = w_probs[labels >= 0, labels]
    # s = s_probs[labels >= 0, labels]
    # plt.plot(y[:300], w[:300].cpu(), color='red')
    # plt.plot(y[:300], s[:300].cpu(), color='blue')
    # plt.show()

    easy_flag = w_mask & s_mask
    select_flag = w_mask + s_mask
    w_selected_flags = w_mask & (~easy_flag)  # H_w
    s_selected_flags = s_mask & (~easy_flag)  # H_s
    mid_flag = w_selected_flags + s_selected_flags  # H

    easy = 0
    mid = 0
    hard = 0
    if cfg.DATASET.TYPE == "cifar10":
        easy_l = [0 for _ in range(0, 10)]
        mid_l = [0 for _ in range(0, 10)]
        hard_l = [0 for _ in range(0, 10)]
        dif_labels = [0 for _ in range(0, 60000)]
        for idx in range(60000):
            if easy_flag[idx]:
                dif_labels[idx] = 0
                easy += 1
                easy_l[labels[idx]] += 1
            elif mid_flag[idx]:
                dif_labels[idx] = 1
                mid += 1
                mid_l[labels[idx]] += 1
            else:
                dif_labels[idx] = 2
                hard += 1
                hard_l[labels[idx]] += 1
    else:
        easy_l = [0 for _ in range(0, 100)]
        mid_l = [0 for _ in range(0, 100)]
        hard_l = [0 for _ in range(0, 100)]
        dif_labels = [0 for _ in range(0, 50000)]
        for idx in range(50000):
            if easy_flag[idx]:
                dif_labels[idx] = 0
                easy += 1
                easy_l[labels[idx]] += 1
            elif mid_flag[idx]:
                dif_labels[idx] = 1
                mid += 1
                mid_l[labels[idx]] += 1
            else:
                dif_labels[idx] = 2
                hard += 1
                hard_l[labels[idx]] += 1

    print('easy:{0}\tmid:{1}\thard:{2}\t'.format(easy, mid, hard))

    model_student.set_diff_level(dif_labels, labels)

    log_path = os.path.join(cfg.LOG.PREFIX, experiment_name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    with open(os.path.join(log_path, 'dif_labels.csv'), 'w') as f:
        w = csv.writer(f)
        w.writerow(dif_labels)
    with open(os.path.join(log_path, 'labels.csv'), 'w') as f:
        w = csv.writer(f)
        w.writerow(labels.cpu().numpy())

    # if cfg.DISTILLER.TYPE == "CRD":
    #     distiller = distiller_dict[cfg.DISTILLER.TYPE](
    #         model_student, model_teacher, cfg, num_data
    #     )
    # else:
    #     distiller = distiller_dict[cfg.DISTILLER.TYPE](
    #         model_student, model_teacher, cfg
    #     )
    #
    # distiller = torch.nn.DataParallel(distiller.cuda())

    # train
    trainer = trainer_dict[cfg.SOLVER.TRAINER](
        experiment_name, model_teacher, model_student, train_loader_weak, train_loader_strong, val_loader, cfg
    )
    trainer.train(resume=resume)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("training for knowledge distillation.")
    parser.add_argument("--cfg", type=str,
                        default="/root/code/MLKD/configs/cifar100/dkd/our_MLKD_DKD_res56_r20.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    main(cfg, args.resume, args.opts)
