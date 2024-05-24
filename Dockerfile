FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

# Arguments to build Docker Image using CUDA
ARG USE_CUDA=1

ENV AM_I_DOCKER True
ENV BUILD_WITH_CUDA "${USE_CUDA}"
ENV TORCH_CUDA_ARCH_LIST "${TORCH_ARCH}"
ENV CUDA_HOME /usr/local/cuda-11.4/
RUN cd /root && git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git

RUN apt update && apt install -y --no-install-recommends wget ffmpeg=7:* \
    libsm6=2:* libxext6=2:* git=1:* \
    vim=2:* 
RUN apt clean -y && apt autoremove -y && rm -rf /var/lib/apt/lists/*

WORKDIR /root/Grounded-Segment-Anything
RUN python3 -m pip install --no-cache-dir -e segment_anything

RUN python3 -m pip install --no-cache-dir wheel
RUN python3 -m pip install --no-cache-dir --no-build-isolation -e GroundingDINO

WORKDIR /root
RUN python3 -m pip install --no-cache-dir diffusers[torch]==0.15.1 opencv-python==3.4.18.65 \
    pycocotools==2.0.6 matplotlib==3.5.3 \
    onnxruntime==1.14.1 onnx==1.13.1 ipykernel==6.16.2 scipy gradio openai
RUN python3 -m pip install gdown

# download pre-trained files
WORKDIR /root/Grounded-Segment-Anything
RUN wget --quiet https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
# following models are optional
RUN wget --quiet https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
RUN wget --quiet https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
RUN wget --quiet https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_tiny.pth

RUN wget --quiet https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# following models are optional for HQ models
RUN gdown --fuzzy https://drive.google.com/file/d/11yExZLOve38kRZPfRx_MRxfIAKmfMY47/view?usp=sharing
RUN gdown --fuzzy https://drive.google.com/file/d/1qobFYrI4eyIANfBSmYcGuWRaSIXfMOQ8/view?usp=sharing
RUN gdown --fuzzy https://drive.google.com/file/d/1Uk17tDKX1YAKas5knI4y9ZJCo0lRVL0G/view?usp=sharing

COPY *.sh /root/Grounded-Segment-Anything/
COPY *.py /root/Grounded-Segment-Anything/

# only for development
RUN apt update && apt install -y eog nano
