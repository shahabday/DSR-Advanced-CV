import numpy as np
from IPython.display import HTML as html_print

from tensorflow.keras.layers import Input

def colored_string(s, color='black'):
    return "<text style=color:{}>{}</text>".format(color, s)

def test_anchors(mAnchors):
    
    anchors = mAnchors((512, 512), 
                           np.array([0.75, 0.5, 0.25]), 
                           ratios=np.array([1, 2, 0.5]))
    
    if anchors.shape != (5, 5, 5, 4):
        return html_print(colored_string("IMPLEMENTATION IS NOT CORRECT", "red"))
    else:
        return html_print(colored_string("IMPLEMENTATION IS CORRECT! GOOD JOB!", 'green'))


def TEST_RPN_MODEL(model):
    
    if model.count_params() != 2382893:
        return html_print(colored_string("IMPLEMENTATION IS NOT CORRECT. IT SHOULD HAVE 2382893 PARAMETERS", "red"))
    
    if len(model.outputs) != 2:
        return html_print(colored_string("IMPLEMENTATION IS NOT CORRECT. YOUR MODEL SHOULD HAVE 2 (TWO) OUTPUT LAYERS", "red"))
    else:
        return html_print(colored_string("IMPLEMENTATION IS CORRECT! GOOD JOB!", 'green'))


def TEST_RPN_MODEL_LOSS(model):
    
    if model.loss != {'classification': 'binary_crossentropy', 'location': 'mse'}:
        return html_print(colored_string("IMPLEMENTATION IS NOT CORRECT. YOU SHOULD HAVE TWO LOSSES (binary_crossentropy and mse)", "red"))
    else:
        return html_print(colored_string("IMPLEMENTATION IS CORRECT! GOOD JOB!", 'green'))


def test_ROI_Pooling(RoIPooling):

    feature_map=Input(batch_shape=(None, None, None,512))
    rois=Input(batch_shape=(None, 4)) 
    ind=Input(batch_shape=(None, ),dtype='int32')

    pool = RoIPooling()

    out = pool(feature_map, rois, ind)

    if list(out.shape) != [None, 7, 7, 512]:
        return html_print(colored_string("IMPLEMENTATION IS NOT CORRECT. YOUR OUTPUT SHOULD BE IN THE SHAPE:[None, 7, 7, 512] FOR INPUT OF size=(7, 7)", "red"))
    else:
        return html_print(colored_string("IMPLEMENTATION IS CORRECT! GOOD JOB!", 'green'))

