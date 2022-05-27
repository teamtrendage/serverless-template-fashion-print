# In this file, we define load_model
# It runs once at server startup to load the model to a GPU

# In this example: A Huggingface BERT model
import os
import sys

sys.path.append('/home/umair/Desktop/AxelerateAI/fashion-print-inpainting/models')

from fashion_inpaint import FashionInpaint

def load_model(device):

    # load the model from cache or local file to the CPU    
    fashion_model = FashionInpaint(device=device)

    # transfer the model to the GPU
    # N/A for this example, it's already on the GPU

    # return the callable model
    return fashion_model