# Install the required libraries
pip install -v -q git+https://github.com/huggingface/transformers.git datasets
pip install -q monai
#Patchify to divide large images into smaller patches for training. (Not necessary for smaller images)
pip install patchify
pip install opencv-python
pip install -U matplotlib

echo "alias kk='kill -9 %'" >> ~/.bashrc
