FROM python:3.7

RUN mkdir -p /raaga

COPY ./requirements.txt ./raaga

WORKDIR /raaga

RUN pip install --trusted-host pypi.python.org -r requirements.txt \
    && pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html

COPY . .requirements

EXPOSE 5000