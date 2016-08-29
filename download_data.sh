savedir="kitti_data"
mkdir -p -- "$savedir"
wget https://www.dropbox.com/s/rpwlnn6j39jjme4/kitti_data.zip?dl=0 -O $savedir/prednet_kitti_data.zip
unzip $savedir/prednet_kitti_data.zip -d $savedir
