FROM python:3.8.16-bullseye
ARG DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    wget \
    git \
    build-essential \
    libgl1 \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    unzip \
    ffmpeg

# Set the working directory
WORKDIR /app

# Clone the SadTalker repository
ADD . /app/SadTalker

# Change the working directory to SadTalker
WORKDIR /app/SadTalker

# Install PyTorch with CPU support
RUN pip install torch==1.13.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git"
# Install dlib
RUN pip install dlib-bin

# Install GFPGAN
RUN pip install git+https://github.com/TencentARC/GFPGAN

# Install dependencies from requirements.txt
RUN pip install -r requirements.txt

# Download models using the provided script
RUN chmod +x scripts/download_models.sh && scripts/download_models.sh

EXPOSE 8000

ENTRYPOINT python -m uvicorn server:app --host 0.0.0.0 --port 8000
