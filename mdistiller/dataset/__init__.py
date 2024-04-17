from .cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample
from .cifar10 import get_cifar10_dataloaders, get_cifar10_dataloaders_sample
from .imagenet import get_imagenet_dataloaders, get_imagenet_dataloaders_sample


def get_dataset(cfg, eval=False):
    # if cfg.DATASET.TYPE == "cifar100":
    #     if cfg.DISTILLER.TYPE == "CRD":
    #         train_loader, val_loader, num_data = get_cifar100_dataloaders_sample(
    #             batch_size=cfg.SOLVER.BATCH_SIZE,
    #             val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
    #             num_workers=cfg.DATASET.NUM_WORKERS,
    #             k=cfg.CRD.NCE.K,
    #             mode=cfg.CRD.MODE,
    #         )
    #     else:
    #         train_loader_weak, train_loader_strong, val_loader, num_data = get_cifar100_dataloaders(
    #             batch_size=cfg.SOLVER.BATCH_SIZE,
    #             val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
    #             num_workers=cfg.DATASET.NUM_WORKERS,
    #         )
    #     num_classes = 100
    # elif cfg.DATASET.TYPE == "imagenet":
    #     if cfg.DISTILLER.TYPE == "CRD":
    #         train_loader, val_loader, num_data = get_imagenet_dataloaders_sample(
    #             batch_size=cfg.SOLVER.BATCH_SIZE,
    #             val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
    #             num_workers=cfg.DATASET.NUM_WORKERS,
    #             k=cfg.CRD.NCE.K,
    #         )
    #     else:
    #         train_loader, val_loader, num_data = get_imagenet_dataloaders(
    #             batch_size=cfg.SOLVER.BATCH_SIZE,
    #             val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
    #             num_workers=cfg.DATASET.NUM_WORKERS,
    #         )
    #     num_classes = 1000
    # else:
    #     raise NotImplementedError(cfg.DATASET.TYPE)

    if eval and cfg.DATASET.TYPE == "cifar100":
        train_loader = get_cifar100_dataloaders(
            batch_size=cfg.SOLVER.BATCH_SIZE,
            val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
            num_workers=4,
            eval=True
        )
        return train_loader
    elif eval and cfg.DATASET.TYPE == "cifar10":
        train_loader = get_cifar10_dataloaders(
            batch_size=cfg.SOLVER.BATCH_SIZE,
            val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
            num_workers=4,
            eval=True
        )
        return train_loader

    if cfg.DATASET.TYPE == "cifar100" and not cfg.DISTILLER.TYPE == "CRD":
        train_loader_weak, train_loader_strong, val_loader, num_data = get_cifar100_dataloaders(
            batch_size=cfg.SOLVER.BATCH_SIZE,
            val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
            num_workers=4,
        )
        num_classes = 100
    elif cfg.DATASET.TYPE == "cifar10" and not cfg.DISTILLER.TYPE == "CRD":
        train_loader_weak, train_loader_strong, val_loader, num_data = get_cifar10_dataloaders(
            batch_size=cfg.SOLVER.BATCH_SIZE,
            val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
            num_workers=4,
        )
        num_classes = 10
    else:
        raise NotImplementedError(cfg.DATASET.TYPE)

    return train_loader_weak, train_loader_strong, val_loader, num_data, num_classes
