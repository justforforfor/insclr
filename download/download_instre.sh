mkdir -p datasets/instre && cd datasets/instre
wget http://123.57.42.89/Dataset_ict/INSTRE/INSTRE_release.rar
unrar x INSTRE_release.rar
mv INSTRE_release/* ./
rm -rf INSTRE_release
wget https://cmp.felk.cvut.cz/~iscenahm/files/test_script.zip
unzip test_script.zip
mv ./test_script/gnd_instre.mat ./
