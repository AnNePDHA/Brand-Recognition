# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.9

EXPOSE 5000

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

###################################
RUN apt-get update && apt-get install -y wget build-essential

RUN wget -O - https://www.openssl.org/source/openssl-1.1.1u.tar.gz | tar zxf -
WORKDIR /openssl-1.1.1u
RUN ./config --prefix=/usr/local
RUN make -j $(nproc)
RUN make install_sw install_ssldirs
RUN ldconfig -v
ENV SSL_CERT_DIR=/etc/ssl/certs
###################################


# Install pip requirements
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

RUN pip install --upgrade pip
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

# RUN python -m pip install tensorflow

WORKDIR /app
COPY . /app
    
# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["python", "app.py"]
