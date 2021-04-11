#! /bin/bash
mkdir -p pretrained/

# install target aravec files
curl https://bakrianoo.s3-us-west-2.amazonaws.com/aravec/full_uni_sg_300_twitter.zip --output pretrained/aravec.zip
unzip pretrained/aravec.zip -d pretrained && rm pretrained/aravec.zip

# install arabic fasttext
curl https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ar.300.bin.gz --output pretrained/cc.ar.300.bin.gz
gunzip -c pretrained/cc.ar.300.bin.gz > pretrained/cc.ar.300.bin && rm pretrained/cc.ar.300.bin.gz

# install elmo
curl http://vectors.nlpl.eu/repository/11/136.zip --output pretrained/136.zip
unzip pretrained/136.zip -d pretrained && rm pretrained/136.zip

# install python3-dev for fasttext to work
sudo apt-get install python3-dev
