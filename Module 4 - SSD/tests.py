import numpy as np
from IPython.display import HTML as html_print

from tensorflow.keras.layers import Input

def colored_string(s, color='black'):
    return "<text style=color:{}>{}</text>".format(color, s)


def test_process_img(p_img):
    
    img = p_img("images/download.png", 224)
    
    if img.shape != (224, 224, 3):
        return html_print(colored_string("IMPLEMENTATION IS NOT CORRECT. CHECK OUT THE RESIZE PART", "red"))
    
    if np.max(img) > 1.0 or np.min(img) < 0.0:
        return html_print(colored_string("IMPLEMENTATION IS NOT CORRECT. IMAGE SCALING IS NOT GOOD, SHOULD BE BETWEEN 0 and 1", "red"))
    else:
        return html_print(colored_string("IMPLEMENTATION IS CORRECT! GOOD JOB!", 'green'))


def test_conv_block(conv_block):
    
    c_block = conv_block(32, Input(shape = (224, 224, 3)))
    
    if "dropout" not in c_block.name :
        return html_print(colored_string("IMPLEMENTATION IS NOT CORRECT. YOUR CONV BLOCK SHOULD END WITH DROPOUT", "red"))
    
    if list(c_block.shape) != [None, 224, 224, 32]:
        return html_print(colored_string("IMPLEMENTATION IS NOT CORRECT", "red"))
    else:
        return html_print(colored_string("IMPLEMENTATION IS CORRECT! GOOD JOB!", 'green'))


def test_residual_block(res_block):
    
    c_block = res_block(16, Input(shape = (224, 224, 3)))
    
    if "concatenate" not in c_block.name :
        return html_print(colored_string("IMPLEMENTATION IS NOT CORRECT. YOUR CONV BLOCK SHOULD END WITH CONCATENATE", "red"))
    
    if list(c_block.shape) != [None, 224, 224, 19]:
        return html_print(colored_string("IMPLEMENTATION IS NOT CORRECT", "red"))
    else:
        return html_print(colored_string("IMPLEMENTATION IS CORRECT! GOOD JOB!", 'green'))