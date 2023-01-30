FROM dkimg/opencv:4.7.0-ubuntu

WORKDIR /root/
RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.8/julia-1.8.5-linux-x86_64.tar.gz
RUN tar zxvf julia-1.8.5-linux-x86_64.tar.gz

ENV PATH="$PATH:/root/julia-1.8.5/bin"

# Uncomment to install shotwell, an image utility. Not required.
#RUN apt-get update && apt-get install shotwell --no-install-recommends -y

# RUN julia -e "using Pkg; Pkg.instantiate()"

RUN apt-get update && apt-get install vim -y

RUN julia -e "using Pkg; Pkg.add(\"PythonCall\")"

CMD "/bin/bash"
