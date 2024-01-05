import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

import logging
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

from methods.base import BaseLearner
from utils.toolkit import tensor2numpy, accuracy_domain

from models.clip_prefix_one_prompt_tuning.net import PrefixOnePromptNet

class PrefixPromptTuning(BaseLearner):

    def __init__(self, args):
        super().__init__(args)
        self._network = PrefixOnePromptNet(args)
        
        self.args = args
        self.query_type = args["query_type"]
        self.EPSILON = args["EPSILON"]
        self.init_epoch = args["init_epoch"]
        self.init_lr = args["init_lr"]
        self.init_lr_decay = args["init_lr_decay"]
        self.init_weight_decay = args["init_weight_decay"]
        self.epochs = args["epochs"]
        self.lrate = args["lrate"] 
        self.lrate_decay = args["lrate_decay"]
        #self.batch_size = args["batch_size"] 
        self.weight_decay = args["weight_decay"] 
        self.num_workers = args["num_workers"] 
        self.knn_k=args["knn_k"]
        self.topk = 1  # 
        self.class_num = self._network.class_num #
        self.all_keys = []

    def after_task(self):
        #self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        
        # logging.info('Exemplar size: {}'.format(self.exemplar_size))
    def begin_incremental(self, data_manager):
        self._cur_task += 1 
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))
    def incremental_train(self, data_manager):
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train', mode='train')
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus) 
        self._train(self.train_loader, self.test_loader)
        try:
            if self._network.module.prefix_prompt is not None:
                self._network.module.prefix_prompt.process_task_count()
        except:
            if self._network.prefix_prompt is not None:
                self._network.prefix_prompt.process_task_count()
        self._network.update_fc()
        if self.query_type=='share_p_query':
            self.shareP_clustering(data_manager)
        elif self.query_type=='vit_query':
            self.vit_clustering(self.train_loader)
        else:
            return
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module


    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        # if self._old_network is not None:
        #     self._old_network.to(self._device)
        paramGrad=0
        for name, param in self._network.named_parameters():
            param.requires_grad_(False)
            if len(self._multiple_gpus) > 1:
                numtask=self._network.module.numtask
            else:
                numtask=self._network.numtask

            if "classifier_pool" + "." + str(numtask) in name:
                param.requires_grad_(True)
                paramGrad+=param.numel() 
            if "share_prompt" in name or "prefix_prompt" in name:
                param.requires_grad_(True)
                paramGrad+=param.numel()

        enabled = set()
        for name, param in self._network.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled},count:{paramGrad}")

        if self._cur_task==0: 
            optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.init_lr, weight_decay=self.init_weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=self.init_epoch)
            self.run_epoch = self.init_epoch
            self.train_function(train_loader,test_loader,optimizer,scheduler)
        else:
            optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.lrate, weight_decay=self.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=self.epochs)
            self.run_epoch = self.epochs
            self.train_function(train_loader, test_loader, optimizer, scheduler)


    def train_function(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.run_epoch)) 
        for _, epoch in enumerate(prog_bar):
            self._network.eval() 
            losses = 0.
            correct, total = 0, 0

            for i, (_, inputs, targets) in enumerate(train_loader): 

                inputs, targets = inputs.to(self._device), targets.to(self._device)

                mask = (targets >= self._known_classes).nonzero().view(-1)

                inputs = torch.index_select(inputs, 0, mask) 

                targets = torch.index_select(targets, 0, mask) - self._known_classes

                logits = self._network(inputs)['logits']

                loss = F.cross_entropy(logits, targets)

                optimizer.zero_grad()
                loss.backward() 
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step() 
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            test_acc = self._compute_accuracy_domain(self._network, test_loader)

            info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                self._cur_task, epoch + 1, self.run_epoch, losses / len(train_loader), train_acc, test_acc)
            prog_bar.set_description(info)

        logging.info(info)

    
    def shareP_clustering(self,data_manager):
        self.all_keys = []
        for task in range(self._cur_task+1):
            features = []
            train_dataset = data_manager.get_dataset(np.arange(self.class_num*task, self.class_num*(task+1)), source='train', mode='train')

            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                with torch.no_grad():
                    if isinstance(self._network, nn.DataParallel):
                        feature = self._network.module.extract_share_prompt_vector(inputs)
                    else:
                        feature = self._network.extract_share_prompt_vector(inputs)
            
                features.append(feature)
            features = torch.cat(features, 0).cpu().detach().numpy()
            clustering = KMeans(n_clusters=self.knn_k, random_state=0).fit(features)
            self.all_keys.append(torch.tensor(clustering.cluster_centers_).to(feature.device))

    def vit_clustering(self, dataloader):
        features = []
        for i, (_, inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            mask = (targets >= self._known_classes).nonzero().view(-1)
            inputs = torch.index_select(inputs, 0, mask)
            with torch.no_grad():
                
                if isinstance(self._network, nn.DataParallel):
                    feature = self._network.module.extract_vector(inputs)
                else:
                    feature = self._network.extract_vector(inputs)
            feature = feature / feature.norm(dim=-1, keepdim=True)
            features.append(feature)
        features = torch.cat(features, 0).cpu().detach().numpy()
        clustering = KMeans(n_clusters=self.knn_k, random_state=0).fit(features)
        self.all_keys.append(torch.tensor(clustering.cluster_centers_).to(feature.device))

    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy_domain(y_pred.T[0], y_true, self._known_classes, class_num=self.class_num)
        ret['grouped'] = grouped
        ret['top1'] = grouped['total']
        return ret

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)
            with torch.no_grad():
                if self.query_type=='share_p_query':
                    if isinstance(self._network, nn.DataParallel):
                        feature = self._network.module.extract_share_prompt_vector(inputs)
                    else:
                        feature = self._network.extract_share_prompt_vector(inputs)
                elif self.query_type=='vit_query':
                    if isinstance(self._network, nn.DataParallel):
                        feature = self._network.module.extract_vector(inputs)
                    else:
                        feature = self._network.extract_vector(inputs)
                else:
                    return
                taskselection = []

                for task_centers in self.all_keys:
                    tmpcentersbatch = []
                    for center in task_centers: 
                        tmpcentersbatch.append((((feature - center) ** 2) ** 0.5).sum(1))
                    taskselection.append(torch.vstack(tmpcentersbatch).min(0)[0])
                selection = torch.vstack(taskselection).min(0)[1]

                if isinstance(self._network, nn.DataParallel):
                    outputs = self._network.module.interface(inputs, selection)
                else:
                    outputs = self._network.interface(inputs, selection)

            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  

    def _compute_accuracy_domain(self, model, loader):
        
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)['logits']

            predicts = torch.max(outputs, dim=1)[1]
            correct += ((predicts % self.class_num).cpu() == (targets % self.class_num)).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)
