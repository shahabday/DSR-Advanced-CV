import cv2
import numpy as np
from IPython.display import HTML as html_print


def colored_string(s, color='black'):
    return "<text style=color:{}>{}</text>".format(color, s)


def IoUTest(IoU):
    
    boxA = {"x1":100, "x2":120, "y1":320, "y2":440}
            
    boxB = {"x1":100, "x2":120, "y1":320, "y2":440}
    
    if IoU(boxA, boxB) != 1.0:
        return html_print(colored_string("TEST 1 FAILED: IMPLEMENTATION IS NOT CORRECT", "red"))
    
    boxA = {"x1":100, "x2":100, "y1":320, "y2":320}
            
    boxB = {"x1":99, "x2":100, "y1":319, "y2":320}
    
    if IoU(boxA, boxB) != 0.25:
        return html_print(colored_string("TEST 2 FAILED: IMPLEMENTATION IS NOT CORRECT", "red"))
    else:
        return html_print(colored_string("IMPLEMENTATION IS CORRECT! GOOD JOB!", 'green'))

    
def selective_search_generator_TEST(ssearch):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    
    jpg = cv2.imread("images/test.jpg")
    
    obj = {'objects': [{'x1': 439, 'x2': 475, 'y1': 537, 'y2': 574},
                      {'x1': 156, 'x2': 251, 'y1': 277, 'y2': 376},
                      {'x1': 480, 'x2': 524, 'y1': 480, 'y2': 526},
                      {'x1': 763, 'x2': 860, 'y1': 39, 'y2': 136},
                      {'x1': 50, 'x2': 123, 'y1': 318, 'y2': 392},
                      {'x1': 568, 'x2': 621, 'y1': 324, 'y2': 376},
                      {'x1': 375, 'x2': 420, 'y1': 134, 'y2': 182},
                      {'x1': 490, 'x2': 531, 'y1': 486, 'y2': 528},
                      {'x1': 245, 'x2': 317, 'y1': 493, 'y2': 567},
                      {'x1': 859, 'x2': 909, 'y1': 59, 'y2': 111},
                      {'x1': 839, 'x2': 905, 'y1': 152, 'y2': 221}]}
    
    obj['img'] = jpg
    
    result = ssearch(obj, ss, number_of_samples=1, number_of_regions=2000)
    
    if result[0].shape != (2, 224, 224, 3):
        return html_print(colored_string("IMPLEMENTATION IS NOT CORRECT", "red"))
    
    if len(result[1]) != 2:
        return html_print(colored_string("IMPLEMENTATION IS NOT CORRECT", "red"))
    else:
        return html_print(colored_string("IMPLEMENTATION IS CORRECT! GOOD JOB!", 'green'))