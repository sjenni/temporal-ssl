from datasets import convert_ucf101, convert_hmdb51
from constants import UCF101_DIR, UCF101_TFDIR, HMDB51_DIR, HMDB51_TFDIR

convert_ucf101.run(UCF101_DIR, UCF101_TFDIR)
convert_hmdb51.run(HMDB51_DIR, HMDB51_TFDIR)
