import os

# Directories for data and logs
DATA_DIR = '/Data'
LOG_DIR = os.path.join(DATA_DIR, 'Logs/VideoSSL')

# Source directories for datasets
UCF101_DIR = os.path.join(DATA_DIR, 'Datasets/UCF-101')
HMDB51_DIR = os.path.join(DATA_DIR, 'Datasets/HMDB51')

# TF-Records
UCF101_TFDIR = os.path.join(DATA_DIR, 'TF_Records/UCF101')
HMDB51_TFDIR = os.path.join(DATA_DIR, 'TF_Records/HMDB51')
