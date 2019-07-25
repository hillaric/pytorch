import torch
from torch.utils import data
from data.dataset import DogCat
import torch.nn as nn
import models
import torch.optim as optim
import torchnet.meter as meter
from config import DefalutConfig

# main函数
def train():
    # 获取数据
    dataset = DogCat('./data/dogcat/', train=True)
    train_data = data.Dataloader(dataset, batch_size=10, shuffle=True, num_workers=0)
    dataset = DogCat('./data/dogcat/', train=False)
    test_data = data.Dataloader(dataset, batch_size=10, shuffle=True, num_workers=0)
    
    # 定义模型
    model = getattr(models, 'AlexNet')
    if load_model_path:
        model.load(load_model_path)
    if use_gpu:
        model.cuda()
        
    # 定义损失函数以及优化函数
    criterion = nn.CrossEntropyLoss()
    lr = opt.lr
    weight_decay = opt.weight_decay
    optimizer = optim.SGD(model.parameters(),
                         lr,
                         weight_decay)
    
    # 定义评估函数
    loss_meter = meter.AverageValueMeter()
    previous_loss = opt.previous_loss
    
    # 开始训练
    epochs = opt.epochs
    for epoch in range(epochs):
        loss_meter.reset()
        
        # 载入训练数据        
        for i, (data, label) in enumerate(train_data):
            data = Variable(data)
            label = Variable(label)
            if use_gpu:
                data.cuda()
                label.cuda()
            optimizer.zero_grad()
            score = model(data)
            loss = criterion(label, score)
            loss.backward()
            optimizer.step()
        # 更新评估函数
            loss_meter.add(loss.data[0])
        # 可视化模块，略去
            if i % 200 == 0:
                print('batch{},epoch is {}, loss is {}'.format(i, epoch, loss_meter))
        model.save()
        
        # 计算在验证集上的指标以及可视化
        loss = val()
        
        # 可视化略
        
        # 如果损失不再下降，那么将学习率变小
        if loss_meter.value[0] > previous_loss:
            lr = lr * weight_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        previous_loss = loss_meter.value[0]
            
            
            
