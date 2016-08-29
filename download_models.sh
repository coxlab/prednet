savedir="model_data"
mkdir -p -- "$savedir"
wget https://www.dropbox.com/s/f5cn9nsqg9gxd6e/prednet_kitti_model.zip?dl=0 -O $savedir/prednet_kitti_model.zip
unzip $savedir/prednet_kitti_model.zip -d $savedir
wget https://www.dropbox.com/s/whcnajvnhwi2lxp/prednet_kitti_model-extrapfinetuned.zip?dl=0 -O $savedir/prednet_kitti_model-extrapfinetuned.zip
unzip $savedir/prednet_kitti_model-extrapfinetuned.zip -d $savedir
