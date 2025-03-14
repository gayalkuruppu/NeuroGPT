python3 ../src/train_gpt.py --training-steps=50000 --eval_every_n_steps=1000 --log-every-n-steps=3000 \
    --per-device-training-batch-size=32 --per-device-validation-batch-size=32 --num-workers=16 \
    --num_chunks=32 --chunk_len=500 --chunk_ovlp=50 --num-hidden-layers=6 --num-encoder-layers=6 \
    --run-name='32clen2_embed1024' --training-style='CSM_causal' --embedding-dim=1024 --train-data-path='../../tuh_tensors'
