mkdir -p datasets/roxford5k && cd datasets/roxford5k
wget https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images.tgz
wget http://cmp.felk.cvut.cz/revisitop/data/datasets/roxford5k/gnd_roxford5k.pkl
mkdir images && tar -xvzf oxbuild_images.tgz -C images
