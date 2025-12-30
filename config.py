class Config:
    # 模型配置
    model_name = 'chinese-roberta-wwm-ext'
    hidden_size = 768  # BERT模型隐藏层的大小
    passenger_classes = 10  # 顺风车乘客任务的类别数
    driver_classes = 8  # 顺风车车主任务的类别数
    taxi_passenger_classes = 5  # 打车乘客任务的类别数
    taxi_driver_classes = 6  # 打车车主任务的类别数

    # 训练配置
    learning_rate = 1e-4
    batch_size = 32
    num_epochs = 10
    weight_decay = 0.01  # 权重衰减，防止过拟合
    train_val_split = 0.2  # 训练集与验证集的划分比例

    # 数据处理配置
    max_seq_length = 128  # 文本序列的最大长度
    data_path = 'path/to/your/data.csv'  # 数据文件路径
    preprocessed_data_path = 'path/to/preprocessed/data.pkl'  # 预处理后的数据存储路径

    # 日志配置
    log_level = 'INFO'
    log_file = 'RideMatchML/logs/training.log'  # 日志文件路径

    # 损失函数权重配置（可根据需要调整）
    task_loss_weights = {
        'passenger': 1.0,
        'driver': 1.0,
        'taxi_passenger': 1.0,
        'taxi_driver': 1.0
    }

    # GPU配置（如果使用GPU）
    use_cuda = True  # 是否使用CUDA（GPU加速）
    cuda_device = 0  # CUDA设备号

    # 更多配置...
