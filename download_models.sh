savedir="model_data_keras2"
mkdir -p -- "$savedir"
wget https://www.dropbox.com/s/z7ittwfxa5css7a/model_data_keras2.zip?dl=0 -O $savedir/model_data_keras2.zip
unzip -j $savedir/model_data_keras2.zip -d $savedir
rm $savedir/model_data_keras2.zip
