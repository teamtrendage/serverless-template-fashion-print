# In this file, we define download_model
# It runs during container build time to get model weights locally

import gdown
import zipfile
import os

def exractfiles(file, dest):
    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall(dest)


def download_model():
    lama_url = '1BXg36NmgK3fafLfudQ2MKaH9g56_0HKu'
    detr_url = '1PEMktjhDikSkCgxGd8LvdFQkmEl4YzIe'
    
    # print('Downloading Weights 1.....')
    # url = f'https://drive.google.com/uc?id={lama_url}'
    # gdown.download(url,'./downloaded_data/', quiet=False)  
    
    # print('Downloading Weights 2.....')
    # url = f'https://drive.google.com/uc?id={detr_url}'
    # gdown.download(url,'./downloaded_data/', quiet=False)  
    
    print('Download Completed Successfully!')

    print('Extracting and Moving Weights....')
    exractfiles('./downloaded_data/big-lama.zip', '../lama/')
    exractfiles('./downloaded_data/checkpoints.zip', '../detr/')
    print('Extracting and Moving Weights Completed!')

if __name__ == "__main__":
    download_model()
