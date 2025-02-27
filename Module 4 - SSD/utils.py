import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

def IoU(boxA, boxB):
    #evaluate the intersection points 
    xA = np.maximum(boxA[0], boxB[0])
    yA = np.maximum(boxA[1], boxB[1])
    xB = np.minimum(boxA[2], boxB[2])
    yB = np.minimum(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = np.maximum(0, xB - xA + 1) * np.maximum(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    #compute the union 
    unionArea = (boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return interArea / unionArea

def get_anchor(box, anchor_size):
    """
    This algorithm takes a bounding box and produces anchor boxes that overlap with it.
    To estimate if an anchor box is okay, this algorithm uses IoU metric.
    
    Args:
        box - real target taken from the dataset for example
    """
    best_iou = 0.0 

    matching_anchor  = [0, 0, 0, 0]
    matching_index   = (0, 0)
    i = 0 
    j = 0 
        
    w , h = (1/anchor_size, 1/anchor_size)
    
    for x in np.linspace(0, 1, anchor_size +1)[:-1]:
        j = 0 
        for y in np.linspace(0, 1, anchor_size +1)[:-1]:
            xmin = x 
            ymin = y

            xmax = (x + w) 
            ymax = (y + h) 

            anchor_box = [xmin, ymin, xmax, ymax]
            matching_iou = IoU(box, anchor_box)

            # Checks if the currently choosen Anchor box is the best fit for the input box
            if matching_iou > best_iou:
                matching_anchor = anchor_box
                best_iou = matching_iou
                matching_index = (i, j)
            j += 1
        i+= 1
    return matching_anchor, matching_index

def get_outputs(boxes, anchor_size):
    """
    Use tis function to create targets for the input images
    """
    output = np.zeros((anchor_size, anchor_size, 5))
    for box in boxes:
        if max(box) == 0:
            continue
        _, (i, j) = get_anchor(box, anchor_size)
        output[i,j, :] = [1] + box
    return output


def plot_example(img, boxes, img_size, anchor_size):  

    for i in range(0, anchor_size):
        for j in range(0, anchor_size):
            box = boxes[i, j, 1:] * img_size
            label = boxes[i, j, 0]

            if np.max(box) > 0:
                img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (1, 0, 0), 1)

    plt.axis('off')
    plt.imshow(img)
    plt.show()


#visualize the predicted bounding box
def plot_pred(model, img_id, img_size, anchor_size):
    font = cv2.FONT_HERSHEY_SIMPLEX

    raw = cv2.imread(img_id)[:,:,::-1]

    h, w = (512, 512)

    img = cv2.resize(raw, (img_size, img_size)).astype('float32')
    img = np.expand_dims(img, 0)/255. 

    boxes = model(img).numpy()[0]

    raw = cv2.resize(raw, (w, h))

    for i in range(0, anchor_size):
        for j in range(0, anchor_size):
            box = boxes[i, j, 1:] * w
            lbl = round(boxes[i, j, 0], 2)
            if lbl > 0.5:
                color = [random.randint(0, 255) for _ in range(0, 3)]
                raw = cv2.rectangle(raw, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 3) 
                raw = cv2.rectangle(raw, (int(box[0]), int(box[1])-30), (int(box[0])+70, int(box[1])), color, cv2.FILLED)
                raw = cv2.putText(raw, str(lbl), (int(box[0]), int(box[1])), font, 1, (255, 255, 255), 2)


    plt.axis('off')
    plt.imshow(raw)
    plt.show()