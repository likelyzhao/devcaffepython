#!/user/bin/env python

import caffe
import numpy as np

np.set_printoptions(threshold='nan')
MODEL_FILE = '../Train/Barcode_train_test.prototxt'
SAVEMODEL = 'save.caffemodel'

sf = open('struct.h','w')
sf2 = open('layers.h','w')

###add the c style heads

sf2.write("#include \"MvsBase.h\"\n\nint32_t modelstuct[]={\n")
sf.write("#include \"MvsBase.h\"\n\nint32_t modelweights[]={\n")

net = caffe.Net(MODEL_FILE,SAVEMODEL,caffe.TRAIN)

#print(net.params)
#for type in net._layer_names:
#   print(type)
#for type in net.blobs:
#    print(net.blobs[type])
#for type in net.params:
#    print((type))
NUMBEREACHLINE = 50

DATACODE = "700,"
CONVCODE = "701,"
POOLCODE = "702,"
IPCODE   = "703,"
RELUCODE = "704,"
LAYERENDCODE = "799"
#### saveing input infos 
sf2.write(DATACODE)
sf2.write('%d,' %net.blobs["data"].data.shape[1])
sf2.write('%d,' %net.blobs["data"].data.shape[2])
sf2.write('%d,' %net.blobs["data"].data.shape[3])


for layerName ,layer in zip(net._layer_names,net.layers):

   if "conv" in layerName:
       sf2.write(CONVCODE)
      # sf2.write('\n')
       sf2.write('%d,' %layer.blobs[0].shape[0])
       sf2.write('%d,' %layer.blobs[0].shape[2])
       sf2.write('%d,' %layer.blobs[0].shape[3])
       sf2.write('\n')   
       weight = net.params[layerName][0].data
       bias = net.params[layerName][1].data

       sf.write('\n//'+layerName+'_weight:\n\n')
       weight.shape = (-1,1)
       idx =0
       for w in weight:
         sf.write('%ff, ' %w)
         if (idx == NUMBEREACHLINE):
             idx =0
             sf.write("\n")
         else:
             idx = idx +1


       sf.write('\n\n//' + layerName + '_bias:\n\n')

       bias.shape = (-1,1)
       idx =0
       for b in bias:
         sf.write('%ff, ' %b)
         if (idx == NUMBEREACHLINE):
             idx =0
             sf.write("\n")
         else:
             idx = idx +1

       sf.write('\n\n')
 
   if "pool" in layerName:
       sf2.write(POOLCODE)
       #sf2.write('\n')
       sf2.write('2')
       sf2.write('\n')   

   if "ip" in layerName:
       sf2.write(IPCODE)
       #sf2.write('\n')
 #      print(len(layer.blobs))
       sf2.write('%d,' %layer.blobs[0].shape[0])
  #     sf2.write('%d,' %layer.blobs[0].shape[1])
       sf2.write('\n')   
       weight = net.params[layerName][0].data
       bias = net.params[layerName][1].data

       sf.write('\n//'+layerName+'_weight:\n\n')
       weight.shape = (-1,1)
       idx =0
       for w in weight:
         sf.write('%ff, ' %w)
         if (idx == NUMBEREACHLINE):
             idx =0
             sf.write("\n")
         else:
             idx = idx +1

       sf.write('\n\n//' + layerName + '_bias:\n\n')

       bias.shape = (-1,1)
       idx =0
       for b in bias:
         sf.write('%ff, ' %b)
         if (idx == NUMBEREACHLINE):
             idx =0
             sf.write("\n")
         else:
             idx = idx +1

       sf.write('\n\n')
 
   if "relu" in layerName:
       sf2.write(RELUCODE)
       sf2.write('\n')




#for layer_type in net.params.types():
#    print(layer_type)
#
#or layer_name,param in net.params.iteritems():
#   sf2.write(layer_name)
#   sf2.write('\n')
#   a = param[0].data.shape
#   print(param[0].data.shape)
#   if "conv" in layer_name:
#      sf2.write('%d,' %param[0].data.shape[0])
#      sf2.write('%d,' %param[0].data.shape[2])
#      sf2.write('%d,' %param[0].data.shape[3])
#      sf2.write('\n')   
#for paramName in net.params.keys():
#   weight = net.params[paramName][0].data
#   bias = net.params[paramName][1].data
#    sf.write(paramName) 
#   sf.write('\n')
#   sf.write('\n'+paramName+'_weight:\n\n')
#   weight.shape = (-1,1)
#   for w in weight:
#       sf.write('%ff, ' %w)
#    sf.write('\n\n' + paramName + '_bias:\n\n')
#    bias.shape = (-1,1)
#   for b in bias:
#       sf.write('%ff, ' %b)
#   sf.write('\n\n')
sf2.write(LAYERENDCODE)
sf2.write("};")
sf.write("};")
sf.close
sf2.close
