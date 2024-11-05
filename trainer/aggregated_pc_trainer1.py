import pytorch_lightning as pl
from pytorch_lightning.metrics.functional.classification import iou
from torch.utils.tensorboard import SummaryWriter
import torch
from data_utils import data_map_KITTI360, data_map_ParisLille3D, data_map_Toronto3D
from data_utils.ioueval import iouEval
from data_utils.collations import *
from numpy import inf, pi, cos, array, expand_dims
from functools import partial
from testing import inference

class AggregatedPCTrainer(pl.LightningModule):
    def __init__(self, model, model_head, criterion, train_loader, val_loader, params):
        super().__init__()
        self.model = model
        self.model_head = model_head
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.params = params
        self.writer = SummaryWriter(f'runs/{self.params.dataset_name}/graph/{self.params.checkpoint}_{self.params.percentage_labels}/{self.params.load_epoch}')
        self.iter_log = 100
        self.best_acc = -1.
        self.best_loss = inf
        self.evaluator = iouEval(n_classes=self.params.num_classes, ignore=0)
        self.val_step = 0
        self.train_step = 0

        if self.params.load_checkpoint:
            self.load_checkpoint()
        
        if self.params.dataset_name == "KITTI360":
            self.labels = data_map_KITTI360.labels
        elif self.params.dataset_name == "ParisLille3D":
            self.labels = data_map_ParisLille3D.labels
        elif self.params.dataset_name == "Toronto3D":
            self.labels = data_map_Toronto3D.labels
        
        
    ############################################################################################################################################
    # FORWARD                                                                                                                                 #
    ############################################################################################################################################

    def forward(self, x):
        h = self.model(x)
        z = self.model_head(h)

        return z

    ############################################################################################################################################

    ############################################################################################################################################
    # TRAINING                                                                                                                                 #
    ############################################################################################################################################

    def training_step(self, batch, batch_nb):
        # for downstream task the batch is the sample x and labels y
        self.train_step += 1
        x_coord, x_feats, x_label = batch
        x, y = numpy_to_sparse_tensor(x_coord, x_feats, x_label)

        y = y[:,0]

        z = self.forward(x)
        loss = self.criterion(z, y.long())
        pred = z.max(dim=1)[1]
        correct = pred.eq(y).sum().item()
        correct /= y.size(0)
        batch_acc = (correct * 100.)

        self.downstream_iter_callback(loss.item(), batch_acc, pred, y, x.C, True)
        self.scheduler.step()

        return {'loss': loss, 'acc': batch_acc}

    def training_epoch_end(self, outputs):
        avg_loss = torch.FloatTensor([ x['loss'] for x in outputs ]).mean()
        avg_acc = torch.FloatTensor([ x['acc'] for x in outputs ]).mean()
        epoch_dict = {'avg_loss': avg_loss, 'avg_acc': avg_acc}

    ############################################################################################################################################

    ############################################################################################################################################
    # VALIDATION                                                                                                                               #
    ############################################################################################################################################

    def validation_step(self, batch, batch_nb):
        pass

    def validation_epoch_end(self, outputs):
        if self.current_epoch == self.params.epochs - 1:
            model_acc, model_miou, model_class_iou = inference(self.params.dataset_name, self.params.num_classes, self.model, self.model_head, self.val_loader)
            self.downstream_iter_callback_val(model_acc, model_miou, model_class_iou)
            self.validation_epoch_callback()

        torch.cuda.empty_cache()
    ############################################################################################################################################

    ############################################################################################################################################
    # CALLBACKS                                                                                                                                #
    ############################################################################################################################################

    def downstream_iter_callback(self, batch_loss, batch_acc, pred, target, coord, is_train):
        # after each iteration we log the losses on tensorboard
        self.evaluator.addBatch(pred.long().cpu().numpy(), target.long().cpu().numpy())
        self.evaluator.addLoss(batch_loss)

        if self.train_step % self.iter_log == 0 or not is_train:
            if is_train:
                self.write_summary(
                    'training/learning_rate',
                    self.scheduler.get_lr()[0],
                    self.train_step,
                )

            # loss
            self.write_summary(
                'training/loss' if is_train else 'validation/loss',
                self.evaluator.getloss(),
                self.train_step if is_train else self.val_step,
            )

            # accuracy
            self.write_summary(
                'training/acc' if is_train else 'validation/acc',
                self.evaluator.getacc(),
                self.train_step if is_train else self.val_step,
            )

            # mean iou
            mean_iou, class_iou = self.evaluator.getIoU()
            self.write_summary(
                'training/miou' if is_train else 'validation/miou',
                mean_iou.item(),
                self.train_step if is_train else self.val_step,
            )
            
            # per class iou
            for class_num in range(class_iou.shape[0]):
                self.write_summary(
                    f'training/per_class_iou/{self.labels[class_num]}' if is_train else f'validation/per_class_iou/{self.labels[class_num]}',
                    class_iou[class_num].item(),
                    self.train_step if is_train else self.val_step,
                )
            
            if not is_train:
                return mean_iou, class_iou
            
            self.evaluator.reset()

    def downstream_iter_callback_val(self, model_acc, model_miou, model_class_iou):
        # after each iteration we log the losses on tensorboard

        if self.current_epoch % 1 == 0:
              
            # accuracy
            self.write_summary(
                'validation/acc',
                model_acc,
                self.current_epoch,
            )

            self.write_summary(
                'validation/miou',
                model_miou.item(),
                self.current_epoch,
            )

            # per class iou
            for class_num in range(model_class_iou.shape[0]):
                self.write_summary(
                    f'validation/per_class_iou/{self.labels[class_num]}',
                    model_class_iou[class_num].item(),
                    self.current_epoch,
                )



    def validation_epoch_callback(self):
        self.save_checkpoint(f'lastepoch{self.current_epoch}')

        #self.validation_mesh_writer()


    ############################################################################################################################################

    ############################################################################################################################################
    # SUMMARY WRITERS                                                                                                                          #
    ############################################################################################################################################

    def write_summary(self, summary_id, report, iter):
        self.writer.add_scalar(summary_id, report, iter)

    # def validation_mesh_writer(self):
    #     val_iterator = iter(self.val_loader)

    #     # get just the first iteration(BxNxM) validation set point clouds
    #     x, y = next(val_iterator)
    #     z = self.forward(x)
    #     for i in range(self.params.batch_size):
    #         points = x.C.cpu().numpy()
    #         labels = z.max(dim=1)[1].cpu().numpy()

    #         batch_ind = points[:, 0] == i
    #         points = expand_dims(points[batch_ind][:, 1:], 0) * self.params.sparse_resolution
    #         colors = array([ color_map[lbl][::-1] for lbl in labels[batch_ind] ])
    #         colors = expand_dims(colors, 0)

    #         point_size_config = {
    #             'material': {
    #                 'cls': 'PointsMaterial',
    #                 'size': 0.3
    #             }
    #         }
        
    #         self.writer.add_mesh(
    #             f'validation_vis_{i}/point_cloud',
    #             torch.from_numpy(points),
    #             torch.from_numpy(colors),
    #             config_dict=point_size_config,
    #             global_step=self.current_epoch,
    #         )

    #     del val_iterator

    ############################################################################################################################################

    ############################################################################################################################################
    # CHECKPOINT HANDLERS                                                                                                                      #
    ############################################################################################################################################

    def load_checkpoint(self):
        self.configure_optimizers()

        if self.params.contrastive:
            # load model, best loss and optimizer
            file_name = f'{self.params.log_dir}/pretraining/{self.params.load_epoch}_model_pretraining.pt'
            checkpoint = torch.load(file_name, map_location='cuda:0')
            self.model.load_state_dict(checkpoint['model'])
            print(f'Contrastive {file_name} loaded from epoch {checkpoint["epoch"]}')
        else:
            # load model, best loss and optimizer
            file_name = f'{self.params.log_dir}/lastepoch199_model_segment_contrast.pt'
            checkpoint = torch.load(file_name)
            self.model.load_state_dict(checkpoint['model'])
            self.train_step = checkpoint['train_step']
            self.val_step = checkpoint['val_step']
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            print(f'{file_name} loaded from epoch {checkpoint["epoch"]}')
            
            # load model head
            file_name = f'{self.params.log_dir}/lastepoch199_model_head_segment_contrast.pt'
            checkpoint = torch.load(file_name)
            self.model_head.load_state_dict(checkpoint['model'])

    def save_checkpoint(self, checkpoint_id):
        # save the best loss checkpoint
        print(f'Writing model checkpoint for {checkpoint_id}')
        state = {
            'model': self.model.state_dict(),
            'epoch': self.current_epoch,
            'optimizer': self.optimizer.state_dict(),
            'val_loss': self.best_loss,
            'scheduler': self.scheduler.state_dict(),
            'params': self.params,
            'train_step': self.train_step,
            'val_step': self.val_step,
        }
        file_name = f'{self.params.log_dir}/{self.params.checkpoint}/{self.params.dataset_name}/{self.params.percentage_labels}/{checkpoint_id}_model_{self.params.load_epoch}.pt'

        torch.save(state, file_name)

        state = {
            'model': self.model_head.state_dict(),
            'epoch': self.current_epoch,
            'optimizer': self.optimizer.state_dict(),
            'val_loss': self.best_loss,
            'scheduler': self.scheduler.state_dict(),
            'params': self.params,
            'train_step': self.train_step,
            'val_step': self.val_step,
        }
        file_name = f'{self.params.log_dir}/{self.params.checkpoint}/{self.params.dataset_name}/{self.params.percentage_labels}/{checkpoint_id}_model_head_{self.params.load_epoch}.pt'

        torch.save(state, file_name)

    ############################################################################################################################################

    ############################################################################################################################################
    # OPTIMIZER CONFIG                                                                                                                         #
    ############################################################################################################################################

    def configure_optimizers(self):
        # define optimizers
        if not self.params.linear_eval:
            print('Fine-tuning!')
            optim_params = list(self.model.parameters()) + list(self.model_head.parameters())
        else:
            print('Linear eval!')
            optim_params = list(self.model_head.parameters())
            self.model.eval()

        #optimizer = torch.optim.Adam(optim_params, lr=self.params.lr, weight_decay=self.params.decay_lr)
        optimizer = torch.optim.SGD(
            optim_params, lr=self.params.lr, momentum=0.9, weight_decay=self.params.decay_lr, nesterov=True
        )

        def cosine_schedule_with_warmup(k, num_epochs, batch_size, dataset_size):
            iter_per_epoch = (dataset_size + batch_size - 1) // batch_size
            return 0.5 * (1 + cos(pi * k /
                                    (num_epochs * iter_per_epoch)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=partial(
                cosine_schedule_with_warmup,
                num_epochs=self.params.epochs,
                batch_size=self.params.batch_size,
                dataset_size=len(self.train_loader) * self.params.batch_size,
            )
        )

        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.params.lr, eta_min=self.params.lr / 1000)

        self.optimizer = optimizer
        self.scheduler = scheduler

        return [optimizer], [scheduler]

    def compute_grad(self):
        param_count = 0
        grad_ = 0.0

        # get grad for model parameters
        for f in self.model.parameters():
            param_count += 1
            if f.grad is None:
                continue
            grad_ += torch.sum(torch.abs(f.grad))

        # get grad for head parameters
        for f in self.model_head.parameters():
            param_count += 1
            if f.grad is None:
                continue
            grad_ += torch.sum(torch.abs(f.grad))

        grad_ /= param_count

        return grad_

    ############################################################################################################################################

    #@pl.data_loader
    def train_dataloader(self):
        return self.train_loader

    #@pl.data_loader
    def val_dataloader(self):
        return self.val_loader

    #@pl.data_loader
    def test_dataloader(self):
        pass
