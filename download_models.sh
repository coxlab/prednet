savedir="model_data"
mkdir -p -- "$savedir"
wget https://www.dropbox.com/s/n6hllbbaeh3fpj9/prednet_kitti_model.zip?dl=0 -O $savedir/prednet_kitti_model.zip
unzip $savedir/prednet_kitti_model.zip -d $savedir
wget https://www.dropbox.com/s/zhcp20ixvufnma8/prednet_kitti_model-extrapfinetuned.zip?dl=0 -O $savedir/prednet_kitti_model-extrapfinetuned.zip
unzip $savedir/prednet_kitti_model-extrapfinetuned.zip -d $savedir
