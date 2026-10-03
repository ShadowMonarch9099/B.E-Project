FROM python:3.11-slim

# Tesseract OCR with English, Hindi and Marathi (plan section 6.2 step 2); LibreOffice Writer
# for DOCX -> PDF (section 6.2 step 3) with Devanagari fonts so converted pages render; a C++
# compiler for packages without a wheel on this platform (fasttext on arm64).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       tesseract-ocr tesseract-ocr-hin tesseract-ocr-mar \
       libreoffice-writer-nogui fonts-lohit-deva fonts-lohit-deva-marathi fonts-dejavu-core \
       build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# CPU-only PyTorch: the CUDA build is several GB and the app runs on CPU.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir -r requirements.txt

COPY alembic.ini ./
COPY migrations ./migrations
COPY app ./app
COPY bench ./bench
COPY eval ./eval
COPY tools ./tools

# Models (E5, reranker, IndicLID, IndicXlit, NLLB) download on first start into the data
# volume, so they survive rebuilds.
ENV SAKSHYA_DATA_DIR=/data \
    HF_HOME=/data/hf \
    SAKSHYA_LIBREOFFICE_CMD=/usr/bin/soffice \
    PYTHONUNBUFFERED=1
VOLUME /data
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=10 \
    CMD curl -fsS http://127.0.0.1:8000/api/health > /dev/null || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
