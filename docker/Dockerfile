# pytorchが提供するpytorch latestイメーjiをベースとしてダウンロード
# for cuda under 11.4
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# Docker実行してシェルに入ったときの初期ディレクトリ（ワークディレクトリ）の設定
WORKDIR /root/

# nvidia-container-runtime（描画するための環境変数の設定）
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# install minimum apps
RUN apt update && apt-get -y install git vim less autoconf cmake

# check pytorch version
RUN python -c "import torch; print('using pytorch: ', torch.__version__)"
