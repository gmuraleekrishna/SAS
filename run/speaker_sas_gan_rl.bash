name="Speaker_sas_dist_area_gan_rl"
flag="--angleFeatSize 128
--train ganspeaker
--start_iter 0
--method sas
--encoder solo
--model_name 'Speaker_sas'
--discriminator_type cnn
--activation sign
--speaker ./logs/Speaker_sas_test/state_dict/best_val_unseen_spice
--augment_every 4
--hparams weight_drop=0.95,w_loss_att=0.2,warmup_iter=100,fact_dropout=0.0,top_k_facts=6,top_k_attribs=6,loss_sp=0.9
--subout max --dropout 0.55 --optim adam --lr 0.0005 --iters 120000 --maxAction 35 --featdropout 0.3 --batchSize 16"
mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=./build python3 r2r_src/train.py $flag --name $name
