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
    batch_size = 16
    number_of_epochs = 10
    num_workers = 0

    # Optimizer parameters
    learning_rate = 0.005