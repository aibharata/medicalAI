import pytest
import medicalai as ai
import tensorflow as tf

INPUT_SIZE = (224,224,3)
OUTPUT_SIZE = 3
def test_all_network():
    networks =  [x for x in dir(ai.networks) if '__' not in x and not x.islower()]
    remove = ['Activation','Conv2D','Dense','Dropout','Flatten','MaxPooling2D','Sequential','NetworkInit','inceptionResnet']
    for net in remove:
        networks.remove(net)

    for net in networks:
        print(10*'-','Checking Network Initialization for :',net,10*'-',)
        myNet = ai.networks.get(net)(INPUT_SIZE,OUTPUT_SIZE)
        assert myNet.output.shape[1]==OUTPUT_SIZE , "Network Initialization: Failed For "+net
        print(10*'-',net,': PASSED',10*'-')


def test_tinyMedNet_v2():
    net = 'tinyMedNet_v2'
    convLayers = 3
    print(10*'-','Checking Network Initialization with Parameters for :',net,10*'-',)
    myNet = ai.get(net)(INPUT_SIZE,OUTPUT_SIZE, convLayers=convLayers)
    assert len(myNet.layers) == 6+(convLayers-1)*2, "Network Initialization: Got Wrong Number of Layers - Failed For "+net
    print(10*'-',net,': PASSED',10*'-')
