# Stage 2 AI worker - ONE container, ONE process.
# "python main.py" runs the pipeline and starts the live ingest uploader as a
# background thread, so there is no second process / supervisor to manage.
FROM python:3.12-slim

# Runtime libraries the Python packages need:
#   libgl1 / libglib2.0-0 / libsm6 / libxext6 / libxrender1 -> opencv-python
#   libgomp1                                               -> paddlepaddle (OpenMP)
#   ffmpeg                                                 -> annotated RTSP publish (--publish)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 libsm6 libxext6 libxrender1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# CPU-only torch must be installed first, otherwise `pip install -r
# requirements.txt` would download the multi-GB CUDA build for nothing.
RUN pip install --upgrade pip \
 && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# ---- Offline warm-up -------------------------------------------------------
# PaddleOCR downloads its models on first use. Do that NOW, during the build, so
# the running container needs NO internet. The models are cached under /root
# (outside /app), so the ./:/app bind mount cannot hide them.

# Paddle's oneDNN path crashes on some CPUs for this OCR model build
# ("NotImplementedError: ConvertPirAttribute2RuntimeAttribute ... onednn_instruction.cc"),
# so turn oneDNN off and fall back to the plain CPU kernels instead.
ENV FLAGS_use_mkldnn=0

# Warm up (and verify) the OCR pipeline. Models are cached during init; the
# predict() call is only a smoke test, so if it raises the build still succeeds
# and the log carries the warning.
RUN python -c "import numpy as np; from paddleocr import PaddleOCR; o=PaddleOCR(lang='en', use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False); o.predict(np.zeros((64,256,3), dtype='uint8')); print('PaddleOCR cached + predict OK')" || echo "WARNING: PaddleOCR models cached, but predict failed at build time."

# No display inside a container, so the OpenCV window is always off.
CMD ["python", "main.py", "--headless"]
