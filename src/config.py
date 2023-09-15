class Config:
    # Raw images
    train_image_dir = "./data/train"
    test_image_dir = "./data/test"

    # Annotations
    train_relationship_file = (
        "./data/train-relationships/train_relationships_processed.csv"
    )
    test_relationship_file = "./data/submissions/sample_submission.csv"

    # Dataset/loader parameters
    train_test_split_ratio = 0.8
    
    # Model hyperparameters
    batch_size = 64
    number_of_epochs = 100
    num_workers = 8

    # Optimizer parameters
    learning_rate = 0.008
    momentum = 0.9
