#!/usr/bin/env python

import os.path

# Code to automatically download alexnet from Caffe Model Zoo

url_model_path = "http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel"
name_model = "bvlc_alexnet.caffemodel"
expected_model_SHA1 = "9116a64c0fbe4459d18f4bb6b56d647b63920377"

url_mean_path = "https://github.com/Robert0812/deepsaldet/raw/master/caffe-sal/data/ilsvrc12/imagenet_mean.binaryproto"
name_mean = "imagenet_mean.binaryproto"
expected_mean_SHA1 = "63e4652e656abc1e87b7a8339a7e02fca63a2c0c"

use_urllib = False

try:
    import wget
except ImportError as e:
    print "No python wget.  Using urlib instead.  No status bar for you..."
    print "sudo pip install wget"
    print
    print "The file is about 233M"
    use_urllib = True

def checkSHA1(filepath, expected_sha):
    import hashlib
    sha = hashlib.sha1()
    with open(filepath, 'rb') as f:
        while True:
            block = f.read(2**10) # Magic number: one-megabyte blocks.
            if not block: break
            sha.update(block)
        found_sha = sha.hexdigest()

    if found_sha != expected_sha:
        print "SHA1 does not match expected"
        print
        print "Expected: ",expected_sha
        print "Found:    ",found_sha
    else:
        print "The SHA1 matches!"

def download(url_path,file_name,expected_sha):
    if os.path.isfile(file_name):
        print "File "+file_name+" already exists.  Skipping."
        return

    print "Downloading ",name_model

    if use_urllib:
        import urllib
        urllib.urlretrieve (url_path, file_name)
    else:
        wget.download(url_path)

    print
    print "Checking the SHA..."
    print
    checkSHA1(file_name,expected_sha)


download(url_model_path,name_model,expected_model_SHA1)
download(url_mean_path,name_mean,expected_mean_SHA1)

print
print "Finished"
