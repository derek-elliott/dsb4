FROM marcchpc/pytorch_cuda9:latest

COPY requirements.txt /workspace/requirements.txt

RUN pip install -r requirements.txt

COPY . /workspace

ENTRYPOINT ["python3", "train.py"]
