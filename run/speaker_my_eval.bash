name=speaker
flag="--attn soft --angleFeatSize 128
      --train speaker
      --speaker snap/speaker/state_dict/best_val_unseen_bleu
      --subout max --dropout 0.6 --optim adam --lr 1e-4 --iters 80000 --maxAction 35"
mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=./build python3 r2r_src/my_eval_speaker.py $flag --name $name

# Try this for file logging
# CUDA_VISIBLE_DEVICES=$1 unbuffer python r2r_src/train_assist.py $flag --name $name | tee snap/$name/log
