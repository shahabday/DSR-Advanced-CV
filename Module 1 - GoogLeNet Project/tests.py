from IPython.display import HTML as html_print
from tensorflow.keras.layers import Input, AveragePooling2D, Dense, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.core import Dropout

def colored_string(s, color='black'):
    return "<text style=color:{}>{}</text>".format(color, s)


def TEST_INCEPTIONBLOCK(InceptionBlock):

    if InceptionBlock(Input(shape = (224, 224, 3)), 10, (10, 10), (10, 10), 10).shape[1:] == [224, 224, 40]:
        return html_print(colored_string("IMPLEMENTATION IS CORRECT! GOOD JOB!", 'green'))
    else:
        return html_print(colored_string("IMPLEMENTATION IS NOT CORRECT", "red"))


    
def TEST_AUXILARY(AuxilaryClassifier):
    inputs = Input(shape = (10, 10, 1))
    aux = AuxilaryClassifier(inputs)
    model = Model(inputs=inputs, outputs=aux)
    layers = model.layers
    
    aux_test = True

    if not model.count_params() == 526593:
        aux_test = False
        return html_print(colored_string("IMPLEMENTATION IS NOT CORRECT", "red"))
        
    if not type(layers[1]) == AveragePooling2D:
        aux_test = False
        return html_print(colored_string("FIRST LAYER SHOULD BE AveragePooling2D", "red"))

    if not type(layers[2]) == Conv2D:
        aux_test = False
        return html_print(colored_string("SECOND LAYER SHOULD BE Conv2D", "red"))

    if not type(layers[3]) == Flatten:
        aux_test = False
        return html_print(colored_string("THIRD LAYER SHOULD BE Flatten", "red"))   

    if not type(layers[4]) == Dense:
        aux_test = False
        return html_print(colored_string("FOURTH LAYER SHOULD BE Dense", "red"))

    if not type(layers[5]) == Dropout:
        aux_test = False
        return html_print(colored_string("FIFTH LAYER SHOULD BE Dropout", "red"))

    if not type(layers[6]) == Dense:
        aux_test = False
        return html_print(colored_string("LAST LAYER SHOULD BE Dense", "red"))

    if aux_test:
        return html_print(colored_string("IMPLEMENTATION IS CORRECT! GOOD JOB!", 'green'))


def TEST_GOOGLENET(model):

    if not model.count_params() == 8471347:
        return html_print(colored_string("IMPLEMENTATION IS NOT CORRECT, CHECK YOUR LAYERS AGAIN", "red"))
        
    if not len(model.outputs) == 3:
        return html_print(colored_string("MAKE SURE THAT YOUR OUTPUTS HAVE 3 PARTS. 2 AUXILARY CLASSIFIERS AND FINAL OUTPUT", "red"))

    return html_print(colored_string("IMPLEMENTATION IS CORRECT! GOOD JOB!", 'green'))