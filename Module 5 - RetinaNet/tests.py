import numpy as np
from IPython.display import HTML as html_print

from tensorflow.keras.layers import Input

def colored_string(s, color='black'):
    return "<text style=color:{}>{}</text>".format(color, s)


def TEST_BACKBONE_MODEL(get_backbone):

    bone = get_backbone()
    
    if bone.count_params() != 23587712:
        return html_print(colored_string("IMPLEMENTATION IS NOT CORRECT. IT SHOULD HAVE 2382893 PARAMETERS", "red"))
    
    if len(bone.outputs) != 3:
        return html_print(colored_string("IMPLEMENTATION IS NOT CORRECT. YOUR MODEL SHOULD HAVE 3 (THREE) OUTPUT LAYERS", "red"))
    else:
        return html_print(colored_string("IMPLEMENTATION IS CORRECT! GOOD JOB!", 'green'))


def TEST_HEAD_MODEL(head_model):
    
    head = head_model(10)
    
    if head.count_params() != 2383370:
        return html_print(colored_string("IMPLEMENTATION IS NOT CORRECT. CHECK NUMBER OF CONV LAYERS", "red"))
    else:
        return html_print(colored_string("IMPLEMENTATION IS CORRECT! GOOD JOB!", 'green'))


def TEST_RESNET(model):
    
    out = model(np.random.randn(1, 128, 128, 3))
    
    if list(out.shape) != [1, 3069, 14]:
        return html_print(colored_string("IMPLEMENTATION IS NOT CORRECT. OUTPUT SHOULD BE SHAPE -> [1, 3069, 14], YOUR OUTPUT IS SHAPE -> " + str(list(out.shape)), "red"))
    else:
        return html_print(colored_string("IMPLEMENTATION IS CORRECT! GOOD JOB!", 'green'))