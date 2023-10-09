name="Speaker_sas_dist_area_small"
flag="--angleFeatSize 128
--train speaker
--method sas
--model_name 'Speaker_sas'
--hparams weight_drop=0.95,w_loss_att=0.125,warmup_iter=10000,fact_dropout=0.0,top_k_facts=6,top_k_attribs=6,loss_sp=0.08
--subout max --dropout 0.55 --optim adam --lr 0.0001 --iters 120000 --maxAction 35 --featdropout 0.3 --batchSize 64"
mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=./build python3 r2r_src/train.py $flag --name $name
