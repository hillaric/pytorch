class DefaultConfig(Object):
    env = 'default'
    model = 'AlexNet'
    
    train_data_root = './data/train'
    test_data_root = './data/test1'
    load_model_path = 'checkpoints/model.pth'
    
    batch_size = 128
    use_gpu = True
    num_workers = 4
    print_freq = 20
    debug_file = '/tmp/debug'
    result_file = 'result.csv'
    
    max_epoch = 10
    lr = 0.1
    lr_decay = 0.95
    weight_decay = 1e-4
    
