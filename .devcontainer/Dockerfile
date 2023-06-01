FROM python:3.10

RUN curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m` \
    && chmod +x /usr/local/bin/cog

COPY . /app

WORKDIR /app

RUN pip install git+https://github.com/harmonai-org/sample-generator

RUN pip install k-diffusion v-diffusion-pytorch

# RUN pip install -r requirements.txt || cat "FAIL" > REQUIREMENTS_FAILED.txt

ENTRYPOINT [ "/bin/bash" ] 

