from trainer.aggregated_pc_trainer1 import AggregatedPCTrainer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from utils import *
import argparse
from numpy import inf
from losses.downstream_criterion import *
import MinkowskiEngine as ME

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SparseSimCLR')

    parser.add_argument('--dataset-name', type=str, default='Toronto3D',
                        help='Name of dataset (default: KITTI360')
    parser.add_argument('--data-dir', type=str, default='/home/reza/PHD/Data/Toronto3D/fps_knn',
                        help='Path to dataset (default: ./Datasets/KITTI360')
    parser.add_argument('--num_classes', type=int, default=9,
                        help='Number of classes in the dataset')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input training batch-size')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of training epochs (default: 15)')
    parser.add_argument('--lr', type=float, default=2.4e-1,
                        help='learning rate (default: 2.4e-1')
    parser.add_argument("--decay-lr", default=1e-4, action="store", type=float,
                        help='Learning rate decay (default: 1e-4')
    parser.add_argument('--log-dir', type=str, default='checkpoint',
                        help='logging directory (default: checkpoint)')
    parser.add_argument('--checkpoint', type=str, default='fine_tune',
                        help='model checkpoint (default: classifier_checkpoint)')
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='using cuda (default: True')
    parser.add_argument('--device-id', type=int, default=0,
                        help='GPU device id (default: 0')
    parser.add_argument('--feature-size', type=int, default=128,
                        help='Feature output size (default: 128')
    parser.add_argument('--sparse-resolution', type=float, default=0.05,
                        help='Sparse tensor resolution (default: 0.05')
    parser.add_argument('--percentage-labels', type=float, default=0.01,
                        help='Percentage of labels used for training (default: 1.0')
    parser.add_argument('--num-points', type=int, default=80000,
                        help='Number of points sampled from point clouds (default: 80000')
    parser.add_argument('--sparse-model', type=str, default='MinkUNet',
                        help='Sparse model to be used (default: MinkUNet')
    parser.add_argument('--linear-eval', action='store_true', default=False,
                        help='Fine-tune or linear evaluation (default: False')
    parser.add_argument('--load-checkpoint', action='store_true', default=True,
                        help='load checkpoint (default: True')
    parser.add_argument('--load-epoch', type=str, default='lastepoch199',
                        help='model checkpoint (default: classifier_checkpoint)')
    parser.add_argument('--contrastive', action='store_true', default=True,
                        help='use contrastive pre-trained weights (default: False')
    parser.add_argument('--accum-steps', type=int, default=1,
                        help='Number steps to accumulate gradient')
    parser.add_argument('--pre-training', action='store_true', default=False,
                        help='use points intensity (default: False')
    parser.add_argument('--segment-contrast', action='store_true', default=False,
                        help='Use segments patches for contrastive learning (default: False')
    args = parser.parse_args()



    # checkpoints_num = np.arange(4, 199, 5, dtype=int)
    checkpoints_num = [199]
    for num in checkpoints_num:
        args.load_epoch = "epoch" + str(num)
        
        if args.use_cuda:
            dtype = torch.cuda.FloatTensor
            device = torch.device("cuda")
            print('GPU')
        else:
            dtype = torch.FloatTensor
            device = torch.device("cpu")
    
        set_deterministic()
    
        data_train, data_test = get_dataset(args)
        train_loader, test_loader = get_data_loader(data_train, data_test, args)
    
    
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    
        model = get_model(args, dtype)
        model_head = get_classifier_head(args, dtype)
    
        if torch.cuda.device_count() > 1:
            model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
            model_head = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model_head)
    
            model_agg_pc = AggregatedPCTrainer(model, model_head, criterion, train_loader, test_loader, args)
            trainer = Trainer(gpus=-1, accelerator='ddp', check_val_every_n_epoch=args.epochs, max_epochs=args.epochs, accumulate_grad_batches=args.accum_steps)
            trainer.fit(model_agg_pc)

        else:
            model_agg_pc = AggregatedPCTrainer(model, model_head, criterion, train_loader, test_loader, args)
            trainer = Trainer(gpus=[0], check_val_every_n_epoch=args.epochs, max_epochs=args.epochs, accumulate_grad_batches=args.accum_steps)
            trainer.fit(model_agg_pc)
