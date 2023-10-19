# Install the required libraries
#SAM
pip install git+https://github.com/facebookresearch/segment-anything.git
#Transformers
pip install -q git+https://github.com/huggingface/transformers.git
#Datasets to prepare data and monai if you want to use special loss functions
pip install datasets
pip install -q monai
#Patchify to divide large images into smaller patches for training. (Not necessary for smaller images)
pip install patchify
