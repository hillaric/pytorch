def val(model, test_data):
    # 把模型设置为测试模式
    model.eval()
    
    confusion_matrix = meter.ConfusionMeter(2)
    
    # 把模型回复为训练模式    
    for i,(data, label) in enumerate(test_data):
        data = Variable(data)
        label = Variable(label)
        
        if use_gpu:
            data.cuda()
            label.cuda()
        score = model(data)
        confusion_matrix.add(score.data.squeeze(), label.long())
        
    model.train()
    
    
    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
    return confusion_matrix, accuracy
