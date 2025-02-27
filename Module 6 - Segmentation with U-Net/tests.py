import numpy as np
from IPython.display import HTML as html_print

from tensorflow.keras.layers import Input

def colored_string(s, color='black'):
    return "<text style=color:{}>{}</text>".format(color, s)


def TEST_ENCODER_BLOCK(encoder_block):

    outs = encoder_block(Input(shape=(224, 224, 3)), 32)
    
    if len(list(outs)) != 2:
        return html_print(colored_string("IMPLEMENTATION IS NOT CORRECT. THIS FUNCTION SHOULD RETURN TWO LAYERS", "red"))

    if "max_pooling2d" not in outs[0].name :
        return html_print(colored_string("IMPLEMENTATION IS NOT CORRECT. FIRST OUTPUT SHOULD BE FROM THE MaxPooling2d LAYER", "red"))


    if "conv2d" not in outs[1].name :
        return html_print(colored_string("IMPLEMENTATION IS NOT CORRECT. SECOND OUTPUT SHOULD BE FROM THE Conv2d LAYER", "red"))
    

    if "max_pooling2d" not in outs[0].name :
        return html_print(colored_string("IMPLEMENTATION IS NOT CORRECT. FIRST OUTPUT SHOULD BE FROM THE MaxPooling2d LAYER", "red"))
    else:
        return html_print(colored_string("IMPLEMENTATION IS CORRECT! GOOD JOB!", 'green'))



def TEST_DECODER_BLOCK(decoder_block, encoder_block):
    
    outs = encoder_block(Input(shape=(224, 224, 3)), 32)

    dec = decoder_block(Input(shape=(112, 112, 3)), outs[1], 32)
    
    if "conv2d" not in dec.name :
        return html_print(colored_string("IMPLEMENTATION IS NOT CORRECT. SECOND OUTPUT SHOULD BE FROM THE Conv2d LAYER", "red"))
    
    
    if list(dec.shape) !=  [None, 224, 224, 32]:
        return html_print(colored_string("IMPLEMENTATION IS NOT CORRECT. CHECK LAYERS AGAIN AND MAKE SURE TO USE CONCAT", "red"))
    else:
        return html_print(colored_string("IMPLEMENTATION IS CORRECT! GOOD JOB!", 'green'))


def TEST_UNET_MODEL(model):
    

    if model.count_params() != 1940817:
        return html_print(colored_string("IMPLEMENTATION IS NOT CORRECT.", "red"))

    if len(model.layers) != 41:
        return html_print(colored_string("IMPLEMENTATION IS NOT CORRECT. MODEL SHOULD HAVE 41 LAYERS, IT HAS -> "  + str(len(list(model.layers))), "red"))
    else:
        return html_print(colored_string("IMPLEMENTATION IS CORRECT! GOOD JOB!", 'green'))