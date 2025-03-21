python3 ../src/train_gpt.py --training-style='decoding' --num-decoding-classes=4 \
    --training-steps=10000  --eval_every_n_steps=500 --log-every-n-steps=1000 \
    --num_chunks=2 --per-device-training-batch-size=32 \
    --per-device-validation-batch-size=32 --chunk_len=500 --chunk_ovlp=0 \
    --run-name='dst' --ft-only-encoder='True' --fold_i=0 --num-encoder-layers=6 \
    --num-hidden-layers=6 --learning-rate=1e-4 --use-encoder='True' \
    --embedding-dim=1024  --pretrained-model='/scr/pretrained_models/neuro_gpt/pytorch_model.bin' \
    --dst-data-path="../../bci2a_egg_npz/"
