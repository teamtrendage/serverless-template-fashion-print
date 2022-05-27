# In this file, we define load_model
# It runs once at server startup to load the model to a GPU

import os
import sys

file_path = os.path.abspath(__file__)
dirpath = os.path.dirname(file_path)
dirpath = os.path.dirname(dirpath)
sys.path.append(dirpath)

from models.fashion_inpaint import FashionInpaint

def load_model(device):

    # load the model from cache or local file to the CPU    
    fashion_model = FashionInpaint(device=device)

    # transfer the model to the GPU
    # N/A for this example, it's already on the GPU

    # return the callable model
    return fashion_model
