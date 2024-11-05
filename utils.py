import MinkowskiEngine as ME
import numpy as np
from data_utils.collations import SparseAugmentedCollation, SparseCollation
from data_utils.datasets.AggregatedPCDataLoader import AggregatedPCDataLoader as data_loader
from models.minkunet import *
from models.moco import *
from models.blocks import ProjectionHead, SegmentationClassifierHead

sparse_models = {
    'MinkUNet': MinkUNet,
}

data_class = {
    'KITTI360': 16,
    'ParisLille3D': 10,
    'Toronto3D': 9
}

def set_deterministic():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True

def list_parameters(models):
    optim_params = []
    for model in models:
        optim_params += list(models[model].parameters())

    return optim_params

def get_model(args, dtype):
    return sparse_models[args.sparse_model](in_channels=3,
        out_channels=latent_features[args.sparse_model],
    )#.type(dtype)

def get_projection_head(args, dtype):
    return ProjectionHead(in_channels=latent_features[args.sparse_model], out_channels=args.feature_size)#.type(dtype)

def get_moco_model(args, dtype):
    return MoCo(sparse_models[args.sparse_model], ProjectionHead, dtype, args)

def get_classifier_head(args, dtype):
    if 'UNet' in args.sparse_model:
        return SegmentationClassifierHead(
                in_channels=latent_features[args.sparse_model], out_channels=data_class[args.dataset_name]
            )#.type(dtype)
    else:
        return ClassifierHead(
                in_channels=latent_features[args.sparse_model], out_channels=data_class[args.dataset_name]
            )#.type(dtype)

def get_optimizer(optim_params, args):
    if 'UNet' in args.sparse_model:
        optimizer = torch.optim.SGD(optim_params, lr=args.lr, momentum=0.9, weight_decay=args.decay_lr)
    else:
        optimizer = torch.optim.Adam(optim_params, lr=args.lr, weight_decay=args.decay_lr)

    return optimizer

# def get_class_weights(dataset):
#     weights = list(content.values()) if dataset == 'SemanticKITTI' else list(content_indoor.values())

#     weights = torch.from_numpy(np.asarray(weights)).float()
#     if torch.cuda.is_available():
#         weights = weights.cuda()

#     return weights

def write_summary(writer, summary_id, report, epoch):
    writer.add_scalar(summary_id, report, epoch)

def get_dataset(args):
    percent_labels = 1.0 if args.pre_training else args.percentage_labels
    data_train = data_loader(root=args.data_dir, split='train', dataset_name=args.dataset_name, percentage=percent_labels, 
                                                    pre_training=args.pre_training, resolution=args.sparse_resolution, orig=args.orig)
    if args.pre_training == False:
        data_test = data_loader(root=args.data_dir, split='validation', dataset_name=args.dataset_name, percentage=percent_labels, 
                                                        pre_training=args.pre_training, resolution=args.sparse_resolution, orig=args.orig)

        return data_train, data_test
    else:
        return data_train

def get_data_loader(data_train, data_test, args):
    collate_fn = None

    if args.pre_training:
        collate_fn = SparseAugmentedCollation(args.sparse_resolution, args.num_points, args.segment_contrast)
        train_loader = torch.utils.data.DataLoader(
            data_train,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=0
        )
        return train_loader

    else:
        collate_fn_train = SparseCollation(args.sparse_resolution, 'train', args.num_points)
        if args.inference:
            test_split = 'test'
        else:
            test_split = 'validation'
        collate_fn_test = SparseCollation(args.sparse_resolution, test_split, args.num_points)
        train_loader = torch.utils.data.DataLoader(
            data_train,
            batch_size=args.batch_size,
            collate_fn=collate_fn_train,
            shuffle=True,
            num_workers=0
        )
    
        test_loader = torch.utils.data.DataLoader(
            data_test,
            batch_size=args.batch_size,
            collate_fn=collate_fn_test,
            shuffle=True,
            num_workers=0
        )
    
        return train_loader, test_loader
