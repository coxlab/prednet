savedir="model_data_keras2"
mkdir -p -- "$savedir"
wget https://www.dropbox.com/s/iutxm0anhxqca0z/model_data_keras2.zip?dl=0 -O $savedir/model_data_keras2.zip
unzip $savedir/model_data_keras2.zip -d $savedir
rm $savedir/model_data_keras2.zip
mv $savedir/model_data_keras2/* $savedir
rm -r $savedir/model_data_keras2
