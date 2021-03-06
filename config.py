config = {
    "IMAGE_DIR": "./data/dataset/semantic_drone_dataset/original_images/",
    "MASK_DIR": "./data/dataset/semantic_drone_dataset/label_images_semantic/",
    "num_classes": 23,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "batch_size": 1,
    "lr": 1e-3,
    "num_epochs": 15,
    "weight_decay": 1e-5,
    "backbone": 'resnet18',
    "experiment": "resnet_lr1e-3_wd1e-5_e15",
    "classes": [15, 17],
    "dropout_rate": None
}