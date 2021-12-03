mkdir datasets/gldv2 && cd datasets/gldv2
wget https://s3.amazonaws.com/google-landmark/metadata/train_clean.csv
git clone https://github.com/cvdfoundation/google-landmark.git
mv google-landmark/* ./ && rm -rf google-landmark
./download.sh
