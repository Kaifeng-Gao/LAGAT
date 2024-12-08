import os
import random
import numpy as np
import torch


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.enabled = True   
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False 
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    print(f'CUBLAS_WORKSPACE_CONFIG: {os.environ["CUBLAS_WORKSPACE_CONFIG"]}') # You can print it out to see if the value has been set successfully
    torch.use_deterministic_algorithms(True) 