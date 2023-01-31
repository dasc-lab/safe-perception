FROM dkimg/opencv:4.7.0-ubuntu

WORKDIR /root/
RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.8/julia-1.8.5-linux-x86_64.tar.gz
RUN tar zxvf julia-1.8.5-linux-x86_64.tar.gz

ENV PATH="$PATH:/root/julia-1.8.5/bin"

RUN apt-get update

# Uncomment to install shotwell, an image utility. Not required.
#apt-get install shotwell --no-install-recommends -y

RUN apt-get install vim -y

#RUN julia -e "using Pkg; Pkg.instantiate()"
RUN julia -e "using Pkg; Pkg.add(\"PythonCall\")"

# Needed for OpenCV to run (through PythonCall in Julia)
ENV LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libcurl.so.4"

CMD "/bin/bash"
