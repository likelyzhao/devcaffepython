export PATH=$PATH:/home/huaray/caffe-master/build/tools/

TESTDBDIR=DB/test
TRAINDBDIR=DB/train


rm -r $TESTDBDIR
convert_imageset --gray "" List/test.txt $TESTDBDIR

rm -r $TRAINDBDIR
convert_imageset --gray "" List/train.txt $TRAINDBDIR
