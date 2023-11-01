ROOT_DIR=$1
if [ -z "$ROOT_DIR" ]; then
        ROOT_DIR=$(cd $(dirname $0); pwd)
fi

nvcc $ROOT_DIR/src/main.cpp  $ROOT_DIR/src/topk.cu -o $ROOT_DIR/bin/query_doc_scoring  -I $ROOT_DIR/src -L/usr/local/cuda/lib64 -lcudart -lcuda  -O3 \
        -std=c++14 -Xcompiler -mavx2 \
        -gencode arch=compute_80,code=sm_80 \
        -gencode arch=compute_86,code=sm_86 \
        -gencode arch=compute_86,code=compute_86

echo "build success"