FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

# prepare your environment here

# RUN pip install ...
RUN pip install timm

COPY code /workspace/code


WORKDIR /workspace/code
