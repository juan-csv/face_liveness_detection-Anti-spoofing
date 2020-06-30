import cv2
import f_liveness_detection
import cv2
import numpy as np
import imutils
import time


def bounding_box(img,box,match_name=[]):
    for i in np.arange(len(box)):
        x0,y0,x1,y1 = box[i]
        img = cv2.rectangle(img,
                      (x0,y0),
                      (x1,y1),
                      (0,255,0),3);
        if not match_name:
            continue
        else:
            cv2.putText(img, match_name[i], (x0, y0-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    return img


# inicializar conteo de parpadeos
COUNTER,TOTAL = 0,0
input_type = "webcam"

#----------------------------- Imagen ------------------------------
if input_type == "image":    
    i=2
    list_images = ["none.jpg","juan.jpg","friends4.jpg"]
    im = cv2.imread("data_test/"+list_images[i])
    im = imutils.resize(im, width=720)

    out = f_liveness_detection.detect_liveness(im,COUNTER,TOTAL)
    print(out)

    boxes = out['box_face_frontal']+out['box_orientation']
    tags = out['emotion']+out['orientation']

    res_img = bounding_box(im,boxes,tags)
    cv2.imshow("liveness_detection",res_img)
    cv2.waitKey(0)

#----------------------------- Video ------------------------------
if input_type == "webcam":
    cv2.namedWindow("preview")
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    while True:
        star_time = time.time()
        ret, im = cam.read()
        im = imutils.resize(im, width=720)
        # ingresar flujo de datos
        out = f_liveness_detection.detect_liveness(im,COUNTER,TOTAL)
        boxes = out['box_face_frontal']+out['box_orientation']
        tags = out['emotion']+out['orientation']
        TOTAL= out['total_blinks']
        COUNTER= out['count_blinks_consecutives']
        res_img = bounding_box(im,boxes,tags)

        end_time = time.time() - star_time    
        FPS = 1/end_time
        cv2.putText(res_img,f"FPS: {round(FPS,3)}",(10,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
        cv2.putText(res_img,f"blinks: {round(TOTAL,3)}",(10,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
        cv2.imshow('preview',res_img)
        if cv2.waitKey(1) &0xFF == ord('q'):
            break 