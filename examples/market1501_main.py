from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import os
import numpy as np
import random
import sys
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
sys.path.insert(0, '/mnt/lustre/sunhongbin/person_reid/cuhk03')
from reid import datasets
from reid import models
from reid.dist_metric import DistanceMetric
# from reid.loss import TripletLoss
from reid.trainers import Trainer
from reid.evaluators import Evaluator
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data.sampler import RandomIdentitySampler
# from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint
from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig
# from loss.triplet import TripletLoss
# from loss.MSML import MSMLTripletLoss
from loss.SDML import SDMLTripletLoss

def get_data(name, split_id, data_dir, height, width, batch_size, num_instances,
             workers, combine_trainval):
    root = osp.join(data_dir, name)

    dataset = datasets.create(name, root, split_id=split_id)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = dataset.trainval if combine_trainval else dataset.train
    num_classes = (dataset.num_trainval_ids if combine_trainval
                   else dataset.num_train_ids)

    query = dataset.query
    query_ids = [pid for _, pid, _ in query]
    query_fnames = [fname for fname, _, _ in query]
    query_cams = [cam for _, _, cam in query]
    query_ids_unique = list(set(query_ids))
    query_fnames_new, query_ids_new, query_cams_new = [], [], []
    gallery_fnames_new, gallery_ids_new, gallery_cams_new = [], [], []
    for k in query_ids_unique:
	    idx = query_ids.index(k)
	    query_ids_new.append(k)
	    query_fnames_new.append(query_fnames[idx])
	    query_cams_new.append(query_cams[idx])
	    new_idx = idx + 1
	    while query_cams[idx] == query_cams[new_idx]:
	        new_idx += 1
	    gallery_ids_new.append(k)
	    gallery_fnames_new.append(query_fnames[new_idx])
	    gallery_cams_new.append(query_cams[new_idx])

    query_num = len(query_ids_unique)
    query_test_num = 100  # 2 GPU
    split_num = query_num//query_test_num
    test_set = []
    tmp = []

    for k in range(split_num):
	    for i in range(2):
	        for j in range(k*query_test_num, (k+1)*query_test_num):
		        if i == 0:
		            tmp.extend((query_fnames_new[j], query_ids_new[j], query_cams_new[j]))
		        else:
		            tmp.extend((gallery_fnames_new[j], gallery_ids_new[j], gallery_cams_new[j]))
		        test_set.append(tmp)
		        tmp = []


    train_transformer = T.Compose([
        T.RandomSizedRectCrop(height, width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
    ])

    test_transformer = T.Compose([
        T.RectScale(height, width),
        T.ToTensor(),
        normalizer,
    ])

    train_loader = DataLoader(
        Preprocessor(train_set, root=dataset.images_dir,
                     transform=train_transformer),
        batch_size=batch_size, num_workers=workers,
        sampler=RandomIdentitySampler(train_set, num_instances),
        pin_memory=True, drop_last=True)

    val_loader = DataLoader(
        Preprocessor(dataset.val, root=dataset.images_dir,
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)
    """ 
    test_loader = DataLoader(
        Preprocessor(test_set, root=dataset.images_dir,
		    transform=test_transformer),
        batch_size=2*query_test_num, num_workers=workers,
        shuffle=False, pin_memory=True)
    """
    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)
     
    return dataset, num_classes, train_loader, val_loader, test_loader


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True


    # Create data loaders
    assert args.num_instances > 1, "num_instances should be greater than 1"
    assert args.batch_size % args.num_instances == 0, \
        'num_instances should divide batch_size'
    if args.height is None or args.width is None:
        args.height, args.width = (144, 56) if args.arch == 'inception' else \
                                  (256, 128)
    dataset, num_classes, train_loader, val_loader, test_loader = \
        get_data(args.dataset, args.split, args.data_dir, args.height,
                 args.width, args.batch_size, args.num_instances, args.workers,
                 args.combine_trainval)

    # Create model
    # Hacking here to let the classifier be the last feature embedding layer
    # Net structure: avgpool -> FC(1024) -> FC(args.features)
    model = models.create(args.arch, num_features=1024,
                          dropout=args.dropout, num_classes=args.features)

    # Load from checkpoint
    start_epoch = best_top1 = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        best_top1 = checkpoint['best_top1']
        print("=> Start epoch {}  best top1 {:.1%}"
              .format(start_epoch, best_top1))
    model = nn.DataParallel(model).cuda()

    # Criterion
    criterion = TripletLoss(margin=args.margin).cuda()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    # Schedule learning rate
    def adjust_lr(epoch):
        lr = args.lr if epoch <= 100 else \
            args.lr * (0.001 ** ((epoch - 100) / 50.0))
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)
	    return lr

    title = args.dataset
    log_filename = 'log.txt'
    if not os.path.isdir(args.logs_dir):
        mkdir_p(args.logs_dir)
    logger = Logger(os.path.join(args.logs_dir, log_filename), title=title)
    logger.set_names(['Learning Rate', 'Train Loss', 'Train Prec', 'Test Top1', 'Test Top5', 'Test Top10'])
    # Start training
    for epoch in range(start_epoch, args.epochs):
        lr = adjust_lr(epoch)
        loss, prec = train_model(train_loader, model, optimizer, criterion, epoch)
        top1, top5, top10 = test_model(test_loader, model)
        logger.append([lr, loss, prec, top1, top5, top10])
        is_best = top1 > best_top1
        best_top1 = max(top1, best_top1)
        save_checkpoint({
            'state_dict': model.module.state_dict(),
            'epoch': epoch + 1,
            'best_top1': best_top1,
        }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

        print('\n * Finished epoch {:3d}  top1: {:5.1%}  best: {:5.1%}{}\n'.
              format(epoch+1, top1, best_top1, ' *' if is_best else ''))

    logger.close()
    # Final test
    print('Test with best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])
    top1, top5, top10 = test_model(test_loader, model)


def test_with_trained_model(args):
    # Create data loaders
    assert args.num_instances > 1, "num_instances should be greater than 1"
    assert args.batch_size % args.num_instances == 0, \
        'num_instances should divide batch_size'
    if args.height is None or args.width is None:
        args.height, args.width = (144, 56) if args.arch == 'inception' else \
                                  (256, 128)
    dataset, num_classes, train_loader, val_loader, test_loader = \
        get_data(args.dataset, args.split, args.data_dir, args.height,
                 args.width, args.batch_size, args.num_instances, args.workers, args.combine_trainval)

    model = models.create(args.arch, num_features=1024,
                          dropout=args.dropout, num_classes=args.features)

    model = nn.DataParallel(model).cuda()
    print('Test with best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])
    top1, top5, top10 = test_model(test_loader, model)

def test_with_open_reid(args):
    # Create data loaders
    assert args.num_instances > 1, "num_instances should be greater than 1"
    assert args.batch_size % args.num_instances == 0, \
        'num_instances should divide batch_size'
    if args.height is None or args.width is None:
        args.height, args.width = (144, 56) if args.arch == 'inception' else \
                                  (256, 128)
    dataset, num_classes, train_loader, val_loader, test_loader = \
        get_data(args.dataset, args.split, args.data_dir, args.height,
                 args.width, args.batch_size, args.num_instances, args.workers,
                 args.combine_trainval)
    # Create model
    model = models.create(args.arch, num_features=1024,
                          dropout=args.dropout, num_classes=args.features)

    model = nn.DataParallel(model).cuda()
    print('Test with best model:')
    # checkpoint = load_checkpoint(osp.join(args.logs_dir, 'checkpoint.pth.tar'))
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])
    # Distance metric
    metric = DistanceMetric(algorithm=args.dist_metric)
    # Evaluator
    evaluator = Evaluator(model)
    metric.train(model, train_loader)
    print("Validation:")
    evaluator.evaluate(val_loader, dataset.val, dataset.val, metric)
    print("Test:")
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, metric)


def train_model(train_loader, model, optimizer, criterion, epoch):

    model.train()
    losses = AverageMeter()
    precisions = AverageMeter()

    for batch_idx, inputs in enumerate(train_loader):

        imgs, _, pids, _ = inputs
        inputs = [Variable(imgs)]
        targets = Variable(pids)
        if args.cuda:
            targets = targets.cuda()

        outputs = model(*inputs)
        loss, prec1 = criterion(outputs, targets)

        losses.update(loss.data[0], targets.size(0))
        precisions.update(prec1, targets.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx+1) % 2 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
                epoch, (batch_idx+1), len(train_loader),
                100. * (batch_idx+1) / len(train_loader), loss.data[0]))
            print('Loss {:.3f} ({:.3f})\t'
                  'Prec {:.2%} ({:.2%})\t'
                  .format(losses.val, losses.avg,
                   precisions.val, precisions.avg))
            print()

    return losses.avg, precisions.avg

def test_model(test_loader, model):

    model.eval()
    top1s, top5s, top10s = [], [], []
    for i, (imgs, fnames, pids, cam_ids) in enumerate(test_loader):
	    if args.cuda:
	        imgs = imgs.cuda()
	    imgs = Variable(imgs, requires_grad=False)
	    outputs = model(imgs)
	    outputs = outputs.data.cpu()
	    probe_gallery_feature = torch.split(outputs, outputs.size(0)/2, 0)
	    probe_feature = probe_gallery_feature[0]
	    gallery_feature = probe_gallery_feature[1]
	    score_array = _cmc_test(probe_feature, gallery_feature)

	    top1s.append(score_array[0])
	    top5s.append(score_array[4])
	    top10s.append(score_array[9])

    top1s, top5s, top10s = np.array(top1s), np.array(top5s), np.array(top10s)
    print('Top1(accuracy) : {:.3f}\t''Top5(accuracy) : {:.3f}\t''Top10(accuracy) : {:.3f}'.format(np.mean(top1s), np.mean(top5s), np.mean(top10s)))

    return np.mean(top1s), np.mean(top5s), np.mean(top10s)


def _cmc_test(probe, gallery, rank_max=20):

    num = probe.size(0)
    rank = []
    score = []
    m, n = probe.size(0), gallery.size(0)
    # if args.cuda:
    #     probe, gallery = probe.cuda(), gallery.cuda()
    # probe, gallery = Variable(probe), Variable(gallery)
    dist = torch.pow(probe, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(gallery, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist = dist.addmm_(1, -2, probe, gallery.t())
    dist = dist.clamp(min=1e-12).sqrt()
    dist_value, dist_indices = torch.sort(dist)
    dist_indices_np = dist_indices.numpy()
    for i in range(m):
        dist_idx = dist_indices_np[i]
        if i < 30:
            print(dist_idx[:10])
        for k in range(n):
            if dist_idx[k] == i:
                rank.append(k+1)
                break
    rank_val = 0
    for i in range(rank_max):
        rank_val = rank_val + len([j for j in rank if i == j-1])
        score.append(rank_val / float(m))

    score_array = np.array(score)

    return score_array


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Softmax loss classification")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='cuhk03',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--height', type=int,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int,
                        help="input width, default: 128 for resnet*, "
                             "56 for inception")
    parser.add_argument('--combine-trainval', action='store_true',
                        help="train and val sets together for training, "
                             "val set alone for validation")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 4")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0)
    # loss
    parser.add_argument('--margin', type=float, default=0.5,
                        help="margin of the triplet loss, default: 0.5")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.0002,
                        help="learning rate of all parameters")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=1)
    # metric learning
    parser.add_argument('--dist-metric', type=str, default='euclidean',
                        choices=['euclidean', 'kissme'])
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    # main(args)
    # test_with_trained_model(args)
    test_with_open_reid(args)
