mkdir -p datasets/rparis6k && cd datasets/rparis6k
wget https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_1.tgz
wget https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_2.tgz
wget http://cmp.felk.cvut.cz/revisitop/data/datasets/rparis6k/gnd_rparis6k.pkl
mkdir images && mkdir tmp
tar -xvzf paris_1.tgz -C images && tar -xvzf paris_2.tgz -C tmp
find ./tmp -name "*.jpg" | xargs -I{} mv {} ./images
rm -rf tmp
