# pytorchが提供するpytorch latestイメーjiをベースとしてダウンロード
# for cuda 12.2
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Docker実行してシェルに入ったときの初期ディレクトリ（ワークディレクトリ）の設定
WORKDIR /root/

# nvidia-container-runtime（描画するための環境変数の設定）
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# install minimum apps
RUN apt update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata
RUN apt-get -y install git vim less cmake

# install opencv python
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y && pip install opencv-python

# add useful commands to .bashrc
RUN echo 'alias kk="kill -9 %"' >> ~/.bashrc

