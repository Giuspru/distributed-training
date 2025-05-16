FROM rayproject/ray-ml:2.6.0-py39

RUN pip install --no-cache-dir \
        torchvision==0.18.0 \
        s3torchconnector==1.4.0 \
        minio>=7.2

COPY train_mnist_ray.py /app/train_mnist_ray.py

WORKDIR /app

ENTRYPOINT ["python", "train_mnist_ray.py"]
