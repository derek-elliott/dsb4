FROM marcchpc/pytorch_cuda9:latest

COPY cloud_requirements.txt /workspace/cloud_requirements.txt

RUN pip install -r cloud_requirements.txt

COPY . /workspace

ENTRYPOINT ["python3"]
