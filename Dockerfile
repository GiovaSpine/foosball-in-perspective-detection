FROM python:3.12.11-slim

WORKDIR /app

COPY app_requirements.txt ./

# cv2 required this libraries
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# to speed up installation
RUN pip install --no-cache-dir torch==2.3.1+cpu torchvision==0.18.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

RUN pip install --no-cache-dir -r app_requirements.txt

# app uses utility.py in src
COPY /app .
COPY src ../src  

EXPOSE 5000

CMD ["python", "app.py"]
