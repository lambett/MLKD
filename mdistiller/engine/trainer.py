import os
import time

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import getpass
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import shutil

from .losses import HCR, dkd_loss, loss_fn_kd_crd
from .utils import (
    AverageMeter,
    accuracy,
    validate,
    adjust_learning_rate,
    save_checkpoint,
    load_checkpoint,
    log_msg,
    validate_s,
)


def rename_directory(old_path, new_name):
    # 获取文件夹的父目录
    parent_directory = os.path.dirname(old_path)
    # 构建新的文件夹路径
    new_path = os.path.join(parent_directory, new_name)
    # 判断构建的新文件夹是否存在
    if os.path.exists(new_path):
        # 存在就删除旧的bestacc文件夹
        shutil.rmtree(new_path)
        # 重命名文件夹
        os.rename(old_path, new_path)
    else:
        # 重命名文件夹
        os.rename(old_path, new_path)


class BaseTrainer(object):
    def __init__(self, experiment_name, model_teacher, model_student, train_loader, train_loader_strong, val_loader, cfg):
        self.cfg = cfg
        self.teacher = model_teacher
        self.student = model_student
        self.train_loader = train_loader
        self.train_loader_strong = train_loader_strong
        self.val_loader = val_loader
        self.optimizer = self.init_optimizer(cfg)
        self.best_acc = -1

        self.cov_list = []

        username = getpass.getuser()
        # init loggers
        self.log_path = os.path.join(cfg.LOG.PREFIX, experiment_name)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.tf_writer = SummaryWriter(os.path.join(self.log_path, "train.events"))

    def init_optimizer(self, cfg):
        if cfg.SOLVER.TYPE == "SGD":
            optimizer = optim.SGD(
                self.student.parameters(),
                lr=cfg.SOLVER.LR,
                momentum=cfg.SOLVER.MOMENTUM,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        else:
            raise NotImplementedError(cfg.SOLVER.TYPE)
        return optimizer

    def log(self, lr, epoch, log_dict):
        # tensorboard log
        for k, v in log_dict.items():
            self.tf_writer.add_scalar(k, v, epoch)
        self.tf_writer.flush()
        # wandb log
        if self.cfg.LOG.WANDB:
            import wandb

            wandb.log({"current lr": lr})
            wandb.log(log_dict)
        if log_dict["test_acc"] > self.best_acc:
            self.best_acc = log_dict["test_acc"]
            if self.cfg.LOG.WANDB:
                wandb.run.summary["best_acc"] = self.best_acc
        # worklog.txt
        with open(os.path.join(self.log_path, "worklog.txt"), "a") as writer:
            lines = [
                "-" * 25 + os.linesep,
                "epoch: {}".format(epoch) + os.linesep,
                "lr: {:.2f}".format(float(lr)) + os.linesep,
            ]
            for k, v in log_dict.items():
                lines.append("{}: {:.2f}".format(k, v) + os.linesep)
            lines.append("-" * 25 + os.linesep)
            writer.writelines(lines)

    def train(self, resume=False):
        epoch = 1
        criterion_hcr = HCR()
        criterion_mse = nn.MSELoss()
        # if resume:
        #     state = load_checkpoint(os.path.join(self.log_path, "latest"))
        #     epoch = state["epoch"] + 1
        #     self.distiller.load_state_dict(state["model"])
        #     self.optimizer.load_state_dict(state["optimizer"])
        #     self.best_acc = state["best_acc"]
        while epoch < self.cfg.SOLVER.EPOCHS + 1:
            self.train_epoch(epoch, criterion_hcr, criterion_mse)
            epoch += 1
        print(log_msg("Best accuracy:{}".format(self.best_acc), "EVAL"))
        with open(os.path.join(self.log_path, "worklog.txt"), "a") as writer:
            writer.write("best_acc\t" + "{:.2f}".format(float(self.best_acc)))

        Pd_data_for_covs = pd.DataFrame(self.cov_list)
        # Pd_data_for_covs.to_csv(
        #     "log_cov_cifar100/" + self.cfg.EXPERIMENT.TAG + "_" + str(
        #         time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())) + ".csv", header=False,
        #     index=True)

        if self.cfg.DATASET.TYPE == 'cifar100':
            Pd_data_for_covs.to_csv(
                "log_cov_cifar100/" + self.cfg.EXPERIMENT.TAG + "_" + str(round(float(self.best_acc), 2)) + ".csv",
                header=False,
                index=True)
        else:
            Pd_data_for_covs.to_csv(
                "log_cov_cifar10/" + self.cfg.EXPERIMENT.TAG + "_" + str(round(float(self.best_acc), 2)) + ".csv",
                header=False,
                index=True)

        # 为最后一次进行命名，便于管理离线权重
        rename_directory(self.log_path, os.path.split(self.log_path)[1] + '_' + str(round(float(self.best_acc), 2)))

    def train_epoch(self, epoch, criterion_hcr, criterion_mse):
        lr = adjust_learning_rate(epoch, self.cfg, self.optimizer)
        train_meters = {
            "training_time": AverageMeter(),
            "data_time": AverageMeter(),
            "losses": AverageMeter(),
            "losses_hcr": AverageMeter(),
            "losses_mse": AverageMeter(),
            "top1": AverageMeter(),
            "top5": AverageMeter(),
        }
        num_iter = len(self.train_loader)
        pbar = tqdm(range(num_iter))

        small_cov = 100.0
        train_matrix = None
        batch = self.cfg.SOLVER.BATCH_SIZE

        # train loops
        self.student.train()
        self.teacher.train()
        self.teacher.eval()
        for idx, data in enumerate(zip(self.train_loader, self.train_loader_strong)):
            data, data_strong = data
            msg, covout = self.train_iter(data, data_strong, epoch, train_meters, criterion_hcr, criterion_mse)
            if train_matrix is None:
                train_matrix = covout[:batch]
            else:
                train_matrix = torch.cat((train_matrix, covout[:batch]), dim=0)
            pbar.set_description(log_msg(msg, "TRAIN"))
            pbar.update()
        pbar.close()

        # validate
        test_acc, test_acc_top5, test_matrix = validate_s(self.val_loader, self.student)

        # log
        log_dict = OrderedDict(
            {
                "train_acc": train_meters["top1"].avg,
                "train_loss": train_meters["losses"].avg,
                "test_acc": test_acc,
                "test_acc_top5": test_acc_top5,
            }
        )
        self.log(lr, epoch, log_dict)
        # saving checkpoint
        state = {
            "epoch": epoch,
            "model": self.student.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_acc": self.best_acc,
        }
        student_state = {"model": self.student.state_dict()}
        save_checkpoint(state, os.path.join(self.log_path, "latest"))
        save_checkpoint(
            student_state, os.path.join(self.log_path, "student_latest")
        )
        if epoch % self.cfg.LOG.SAVE_CHECKPOINT_FREQ == 0:
            save_checkpoint(
                state, os.path.join(self.log_path, "epoch_{}".format(epoch))
            )
            save_checkpoint(
                student_state,
                os.path.join(self.log_path, "student_{}".format(epoch)),
            )
        # update the best
        if test_acc >= self.best_acc:
            save_checkpoint(state, os.path.join(self.log_path, "best"))
            save_checkpoint(
                student_state, os.path.join(self.log_path, "student_best")
            )

        cov_start = time.time()
        cov_item = 0
        cov_item = self.cov(train_matrix, test_matrix)
        cov_end = time.time()

        print('cov_item is: ', cov_item, 'cov cost time is ', cov_end - cov_start)
        if small_cov > cov_item:
            small_cov = cov_item
            print('small_cov:', small_cov)
        self.cov_list.append(cov_item)

    def train_iter(self, data, data_strong, epoch, train_meters, criterion_hcr, criterion_mse):
        self.optimizer.zero_grad()
        train_start_time = time.time()
        image, target, index = data
        image_s, target_s, index_s = data_strong
        train_meters["data_time"].update(time.time() - train_start_time)
        image = image.float()
        image = image.cuda(non_blocking=True)
        image_s = image_s.float()
        image_s = image_s.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)

        # forward
        # preds, losses_dict = self.distiller(image=image, target=target, epoch=epoch, criterion_hcr=criterion_hcr)
        feat, out, proj_x, covout = self.student(image)
        # feat_s, out_s, proj_x_s, covout_s = self.student(image_s)
        with torch.no_grad():
            feat_t, logits_teacher, proj_t, proj_logit = self.teacher(image)
            feat_t_s, logits_teacher_s, proj_t_s, proj_logit_s = self.teacher(image_s)
            f_t = proj_t.detach()
            f_t_s = proj_t_s.detach()
        # losses
        if epoch < 160:
            w = 0.00
            w_mse = 0.0
            s_mse = 0.05
        elif epoch < 220:
            w = 0.1
            w_mse = 0.05
            s_mse = 0.1
        else:
            w = 0.2
            w_mse = 0.1
            s_mse = 0.15

        loss_hcr = criterion_hcr.update(logits=out, projections=f_t) * w

        loss_mse = criterion_mse(proj_x, f_t) * w_mse + criterion_mse(proj_x, f_t_s) * s_mse

        loss_ce = self.cfg.DKD.CE_WEIGHT * F.cross_entropy(out, target)
        loss_dkd = min(epoch / self.cfg.DKD.WARMUP, 1.0) * dkd_loss(
            out,
            logits_teacher,
            target,
            self.cfg.DKD.ALPHA,
            self.cfg.DKD.BETA,
            self.cfg.DKD.T,
        )
        loss_multi_kd = loss_fn_kd_crd(out, target, logits_teacher, 0.3, 20)
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_dkd,
            "loss_multi_kd": loss_multi_kd,
            "loss_hcr": loss_hcr,
            "loss_mse": loss_mse
        }

        # backward
        loss = sum([l.mean() for l in losses_dict.values()])
        loss.backward()
        self.optimizer.step()
        train_meters["training_time"].update(time.time() - train_start_time)
        # collect info
        batch_size = image.size(0)
        acc1, acc5 = accuracy(out, target, topk=(1, 5))
        train_meters["losses"].update(loss.cpu().detach().numpy().mean(), batch_size)
        train_meters["losses_hcr"].update(loss_hcr.cpu().detach().numpy().mean(), batch_size)
        train_meters["losses_mse"].update(loss_mse.cpu().detach().numpy().mean(), batch_size)
        train_meters["top1"].update(acc1[0], batch_size)
        train_meters["top5"].update(acc5[0], batch_size)
        # print info
        msg = "Epoch:{}| Time(data):{:.3f}| Time(train):{:.3f}| Loss:{:.4f}| hcr:{:.4f}| mse:{:.4f}| " \
              "Top-1:{:.3f}| Top-5:{:.3f}".format(
            epoch,
            train_meters["data_time"].avg,
            train_meters["training_time"].avg,
            train_meters["losses"].avg,
            train_meters["losses_hcr"].avg,
            train_meters["losses_mse"].avg,
            train_meters["top1"].avg,
            train_meters["top5"].avg,
        )
        return msg, covout

    def cov(self, train_matrix, test_matrix):
        r_shape, e_shape = train_matrix.shape, test_matrix.shape
        train_matrix, test_matrix = train_matrix.view(r_shape[0], -1), \
            test_matrix.view(e_shape[0], -1)
        train_mean, test_mean = torch.mean(train_matrix, dim=0), torch.mean(test_matrix, dim=0)
        tct_matrix = train_matrix[r_shape[0] - e_shape[0]: r_shape[0], :]
        n_dim = train_matrix.shape[1]
        cov_abs = []
        tct_matrix = tct_matrix - train_mean
        test_matrix = test_matrix - test_mean
        for i in range(n_dim):
            rsp_matrix = tct_matrix[:, i].view(e_shape[0], 1)
            mul_mt = rsp_matrix * test_matrix
            cov_ins = torch.sum(mul_mt, dim=0) / (e_shape[0] - 1)
            abs_cov = torch.abs(cov_ins)
            cov_abs.append((torch.sum(abs_cov) / abs_cov.shape[0]).cpu().item())
        return np.sum(cov_abs) / (len(cov_abs))


class CRDTrainer(BaseTrainer):
    def train_iter(self, data, epoch, train_meters):
        self.optimizer.zero_grad()
        train_start_time = time.time()
        image, target, index, contrastive_index = data
        train_meters["data_time"].update(time.time() - train_start_time)
        image = image.float()
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        contrastive_index = contrastive_index.cuda(non_blocking=True)

        # forward
        preds, losses_dict = self.distiller(
            image=image, target=target, index=index, contrastive_index=contrastive_index
        )

        # backward
        loss = sum([l.mean() for l in losses_dict.values()])
        loss.backward()
        self.optimizer.step()
        train_meters["training_time"].update(time.time() - train_start_time)
        # collect info
        batch_size = image.size(0)
        acc1, acc5 = accuracy(preds, target, topk=(1, 5))
        train_meters["losses"].update(loss.cpu().detach().numpy().mean(), batch_size)
        train_meters["top1"].update(acc1[0], batch_size)
        train_meters["top5"].update(acc5[0], batch_size)
        # print info
        msg = "Epoch:{}| Time(data):{:.3f}| Time(train):{:.3f}| Loss:{:.4f}| Top-1:{:.3f}| Top-5:{:.3f}".format(
            epoch,
            train_meters["data_time"].avg,
            train_meters["training_time"].avg,
            train_meters["losses"].avg,
            train_meters["top1"].avg,
            train_meters["top5"].avg,
        )
        return msg
