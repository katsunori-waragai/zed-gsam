FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

# Arguments to build Docker Image using CUDA
ARG USE_CUDA=1

ENV AM_I_DOCKER True
ENV BUILD_WITH_CUDA "${USE_CUDA}"
ENV TORCH_CUDA_ARCH_LIST "8.7+PTX"
ENV CUDA_HOME /usr/local/cuda-11.4/
RUN cd /root && git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git

WORKDIR /root
RUN apt update && apt install -y --no-install-recommends wget ffmpeg=7:* \
    libsm6=2:* libxext6=2:* git=1:* \
    vim=2:* \
    zstd
RUN apt install -y python3-tk
RUN apt clean -y && apt autoremove -y && rm -rf /var/lib/apt/lists/*
# only for development
RUN apt update && apt install -y eog nano

ENV ZED_SDK_INSTALLER=ZED_SDK_Tegra_L4T35.3_v4.1.0.zstd.run
RUN wget --quiet -O ${ZED_SDK_INSTALLER} https://download.stereolabs.com/zedsdk/4.1/l4t35.2/jetsons
RUN chmod +x ${ZED_SDK_INSTALLER} && ./${ZED_SDK_INSTALLER} -- silent

WORKDIR /root/Grounded-Segment-Anything
RUN python3 -m pip install --no-cache-dir -e segment_anything

RUN python3 -m pip install --no-cache-dir wheel
RUN python3 -m pip install --no-cache-dir --no-build-isolation -e GroundingDINO

WORKDIR /root
RUN python3 -m pip uninstall --yes opencv-python-headless opencv-contrib-python
RUN python3 -m pip install --no-cache-dir opencv-python==3.4.18.65 \
    pycocotools==2.0.6 matplotlib==3.5.3 \
    onnxruntime==1.14.1 onnx==1.13.1 scipy mediapipe scikit-image
RUN python3 -m pip install gdown

# download pre-trained files
WORKDIR /root/Grounded-Segment-Anything
RUN wget --quiet https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
# following models are optional
RUN wget --quiet https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
# RUN wget --quiet https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
# RUN wget --quiet https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_tiny.pth

RUN wget --quiet https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# following models are optional for HQ models
# RUN gdown --fuzzy https://drive.google.com/file/d/11yExZLOve38kRZPfRx_MRxfIAKmfMY47/view?usp=sharing
# RUN gdown --fuzzy https://drive.google.com/file/d/1qobFYrI4eyIANfBSmYcGuWRaSIXfMOQ8/view?usp=sharing
# RUN gdown --fuzzy https://drive.google.com/file/d/1Uk17tDKX1YAKas5knI4y9ZJCo0lRVL0G/view?usp=sharing

COPY *.sh *.py setup.cfg /root/Grounded-Segment-Anything/
RUN mkdir -p zedhelper/
RUN mkdir -p tutorial_script/
COPY zedhelper/* /root/Grounded-Segment-Anything/zedhelper/
COPY tutorial_script/*  /root/Grounded-Segment-Anything/tutorial_script/
