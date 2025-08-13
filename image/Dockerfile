FROM nvcr.io/nvidia/pytorch:22.09-py3 

# Install dependencies
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y zip htop screen libgl1-mesa-glx && \
    python -m pip install pyyaml


# Install python dependencies
RUN python -m pip install Flask gevent requests urllib3 && \
    pandas==2.2.0 && \
    networkx==3.2 && \
    matplotlib==3.8.0


RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu117.html

# Install Detectron2
RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Install Mask2Former
RUN git clone https://github.com/facebookresearch/Mask2Former.git && \
    cd Mask2Former && \
    pip install -U opencv-python && \
    pip install git+https://github.com/cocodataset/panopticapi.git && \
    pip install -r requirements.txt && \
    pip install "opencv-python-headless<4.3"

ENV CUDA_HOME='/usr/local/cuda'
ARG FORCE_CUDA=1

RUN cd Mask2Former/mask2former/modeling/pixel_decoder/ops && \
    sh make.sh && \
    mkdir -p /workspace/Mask2Former/data && \
    mkdir -p /workspace/Mask2Former/output && \
    mkdir -p /workspace/Mask2Former/models && \
    mkdir -p /workspace/Mask2Former/auth && \
    mkdir -p /workspace/Mask2Former/image/src

WORKDIR /workspace/Mask2Former
ENV HOME=/workspace/Mask2Former

# add pretrained cfg and weight
COPY models/ /workspace/Mask2Former/models/
COPY auth/ /workspace/Mask2Former/auth/


COPY src/demo_mo_test.py /workspace/Mask2Former/demo
COPY src/data /workspace/Mask2Former/data
COPY src/app.py /workspace/Mask2Former/app.py
COPY src/utils.py /workspace/Mask2Former/utils.py
COPY src/templates /workspace/Mask2Former/templates
COPY src/static /workspace/Mask2Former/static


# copy gcn code
COPY src/preprocessing /workspace/Mask2Former/image/src/preprocessing/
COPY src/bidirected_arcade_scene_graph /workspace/Mask2Former/image/src/bidirected_arcade_scene_graph/
COPY src/GCN_arcade /workspace/Mask2Former/image/src/GCN_arcade/
COPY src/gcn_processor.py /workspace/Mask2Former/image/src/



EXPOSE 7676

# add PYTHONUNBUFFERED environment variable to ensure that Python output is sent straight to the terminal without being buffered
ENV PYTHONUNBUFFERED=1
CMD ["python", "app.py"]