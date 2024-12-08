class Config:
    DATASET_NAME = "yelp"
    TARGET_NODE_TYPE = "review"
    TRAIN_SIZE = 0.4
    VAL_SIZE = 0.1
    RANDOM_SEED = 42
    FORCE_RELOAD = False
    
    EPOCHS = 10000
    TOLERATION = 10
    
    # Model parameters
    IN_CHANNELS = 32
    HIDDEN_CHANNELS = 32
    LABEL_EMBEDDING_DIM = 32
    NUM_LAYERS = 2
    NUM_LABELS = 3
    OUT_CHANNELS = 2
    HEADS = 2
    DROPOUT = 0.6
    
    # Training parameters
    LEARNING_RATE = 0.005
    WEIGHT_DECAY = 5e-4