import numpy as np
import os
import torch
import random
from matplotlib.pylab import plt
from PIL import Image
from torchvision.datasets import ImageFolder
from tqdm.notebook import tqdm
from torchvision import transforms as T
from torch.utils.data import DataLoader
from torch import nn
class minValue():
    def __init__(self, min_num):
        self.min_values = 100000 * np.ones(min_num)
        self.min_indexes= np.array([None]*min_num)

    def update(self, n_value, n_index):
        for i, value in enumerate(self.min_values):
            if n_value <= value:
                if i == len(self.min_indexes) -1 or n_value >= self.min_values[i + 1]:
                    self.min_values[i] = n_value
                    self.min_indexes[i] = n_index
                    break
            else:
                continue

def get_sport_dataloader(batch_size):
    std=torch.Tensor([0.4687, 0.4667, 0.4540])
    mean=torch.Tensor([0.2792, 0.2717, 0.2852])
    t_trans=T.Compose([
                    T.RandomCrop((196,196),padding=4,padding_mode='reflect'),
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomRotation(degrees=(0, 180)),
                    T.ToTensor(),
                    T.Normalize(mean,std)])
    v_trans=T.Compose([
                    T.ToTensor(),
                    T.Normalize(mean,std)])
    train_set=ImageFolder('../dataset/sport_dataset/train',transform=t_trans)
    val_set=ImageFolder('../dataset/sport_dataset/valid',transform=v_trans)
    test_set=ImageFolder("../dataset/sport_dataset/test",transform=v_trans)
    print(len(train_set),len(val_set), len(test_set))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader, train_set.classes

def print_get_acc_loss(epoch, acc_list, loss_list):
    acc = sum(acc_list) / len(acc_list)
    loss = sum(loss_list) / len(loss_list)
    print("Epoch[{}]: vall_acc:{}  val_loss:{}".format(epoch, acc, loss))
    return {'val_loss': loss, 'val_acc': acc}

def acc_cal(preds, y):
    _, output = torch.max(preds, dim=1)
    return torch.sum(y == output) / len(output)

def fit(model, epoch_size, max_lr, train_loader, val_loader, weight_decay=0.01,
        grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    opt = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    lr_schedule = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr, epochs=epoch_size,
                                                      steps_per_epoch=len(train_loader))
    history = []
    for epoch in range(epoch_size):
        acc_list = []
        loss_list = []
        # train
        model.train()

        train_tbar = tqdm(train_loader)
        for x, y in train_tbar:
            loss = model.train_step(x.cuda(), y.cuda())
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            opt.step()
            opt.zero_grad()
            lr_schedule.step()

        val_tbar = tqdm(val_loader)
        for x, y in val_tbar:
            with torch.no_grad():
                values = model.val_step(x.cuda(), y.cuda())

            acc_list.append(values['val_acc'])
            loss_list.append(values['val_loss'])


        history.append(print_get_acc_loss(epoch, acc_list, loss_list))

    return history

def transform_test(img):
    std=torch.Tensor([0.4687, 0.4667, 0.4540])
    mean=torch.Tensor([0.2792, 0.2717, 0.2852])
    v_trans=T.Compose([
                    T.ToTensor(),
                    T.Normalize(mean,std)])
    img= v_trans(img)
    return img.unsqueeze(0)

def make_pred_my(img, model, classes):
  with torch.no_grad():
    probs=model(img.cuda())
  _,pred=torch.max(probs,dim=1)
  pred=classes[pred.item()]
  return {'prediction':pred}
  
def plot_preds_my(model, imgs, classes, path):
    fig=plt.figure(figsize=(20,16))
    for i, img in enumerate(imgs):
      label = path[i].split('/')[-1].replace('.jpg', '')
      i+=1
      pred=make_pred_my(img, model, classes)
      ax = fig.add_subplot(1, 4, i)
      ax.set_xticks([])
      ax.set_yticks([])
      img = img.squeeze()
      plt.imshow(img.permute(1,2,0))
      ax.set_title("label: {}  \npred: {} ".format(label, pred['prediction']))

def denorm(img):
    std=torch.Tensor([0.4687, 0.4667, 0.4540])
    mean=torch.Tensor([0.2792, 0.2717, 0.2852])
    return img*std[0]+mean[0]

def get_class(num, class_dict):
    for index in class_dict:
        if class_dict[index] == int(num):
            return index
    
def plot_preds_knn(path, class_dict, PCA, KNN):
    img1 = np.asarray(Image.open(path[0]).convert('L').resize([224, 224])) / 255.0
    img2 = np.asarray(Image.open(path[1]).convert('L').resize([224, 224])) / 255.0
    img3 = np.asarray(Image.open(path[2]).convert('L').resize([224, 224])) / 255.0
    img4 = np.asarray(Image.open(path[3]).convert('L').resize([224, 224])) / 255.0
    imgs = [img1, img2, img3, img4]
    fig=plt.figure(figsize=(20,16))

    for i, img in enumerate(imgs):
        features = PCA.fit_transform(img)
        pred = KNN.predict(features)
        pred = get_class(pred, class_dict)
        ax = fig.add_subplot(1, 4, i+1)
        ax.set_xticks([])
        ax.set_yticks([])
        img = img.squeeze()
        plt.imshow(Image.open(path[i]).resize([224, 224]))
        label = path[i].split('/')[-1].replace('.jpg','')
        ax.set_title("label: {} \n pred: {} ".format(label, pred))