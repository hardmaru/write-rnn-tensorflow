docker run \
    -it \
    -v $(pwd):/workspace \
    -w /workspace \
    write-rnn-tensorflow:1.5.0-py3 \
    python train.py

