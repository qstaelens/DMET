# Use an official Python base image
FROM python:3.10-slim

# Set environment variables for OpenMP and BLAS
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"
ENV CPATH="/usr/local/include:/usr/include/hdf5/serial:${CPATH}"

# Install system-level dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libomp-dev \
    libhdf5-dev \
    openmpi-bin \
    libopenmpi-dev \
    python3-dev \
    cython3 \
    pkg-config \
    wget \
    && rm -rf /var/lib/apt/lists/*


# Install Python dependencies (PySCF)
RUN pip install --no-cache-dir \
    Cython \
    numpy \
    scipy \
    h5py \
    pyscf \
    block2 \
    matplotlib \
    pandas

# Build and install chemps2 (with MPI support)
WORKDIR /opt
RUN git clone https://github.com/SebWouters/chemps2 && \
    cd chemps2 && mkdir build && cd build && \
    CXX=mpic++ cmake .. \
        -DMKL=OFF \
        -DWITH_MPI=OFF \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DBLAS_LIBRARIES=/usr/lib/x86_64-linux-gnu/libopenblas.so \
        -DLAPACK_LIBRARIES=/usr/lib/x86_64-linux-gnu/liblapack.so \
        -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && make install

# Build and install PyCheMPS2 (Python interface)
WORKDIR /opt/chemps2/PyCheMPS2
RUN python setup.py build_ext -L /usr/local/lib && \
    python setup.py install --prefix=/usr/local

# Copy user application
WORKDIR /app
COPY . /app

COPY lib/libqcdmet.so /usr/local/lib/
RUN ldconfig /usr/local/lib

# Default command
CMD ["python", "-u", "Fe1.py"]
