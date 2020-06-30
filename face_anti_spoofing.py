import random 
import cv2
import imutils
import f_liveness_detection
import questions

# instanciar camara
cv2.namedWindow('liveness_detection')
cam = cv2.VideoCapture(0)

# parameters 
COUNTER, TOTAL = 0,0
counter_ok_questions = 0
counter_ok_consecutives = 0
limit_consecutives = 3
limit_questions = 6
counter_try = 0
limit_try = 50 



def show_image(cam,text,color = (0,0,255)):
    ret, im = cam.read()
    im = imutils.resize(im, width=720)
    #im = cv2.flip(im, 1)
    cv2.putText(im,text,(10,50),cv2.FONT_HERSHEY_COMPLEX,1,color,2)
    return im


for i_questions in range(0,limit_questions):
    # genero aleatoriamente pregunta
    index_question = random.randint(0,5)
    question = questions.question_bank(index_question)
    
    im = show_image(cam,question)
    cv2.imshow('liveness_detection',im)
    if cv2.waitKey(1) &0xFF == ord('q'):
        break 

    for i_try in range(limit_try):
        # <----------------------- ingestar data 
        ret, im = cam.read()
        im = imutils.resize(im, width=720)
        im = cv2.flip(im, 1)
        # <----------------------- ingestar data 
        TOTAL_0 = TOTAL
        out_model = f_liveness_detection.detect_liveness(im,COUNTER,TOTAL_0)
        TOTAL = out_model['total_blinks']
        COUNTER = out_model['count_blinks_consecutives']
        dif_blink = TOTAL-TOTAL_0
        if dif_blink > 0:
            blinks_up = 1
        else:
            blinks_up = 0

        challenge_res = questions.challenge_result(question, out_model,blinks_up)

        im = show_image(cam,question)
        cv2.imshow('liveness_detection',im)
        if cv2.waitKey(1) &0xFF == ord('q'):
            break 

        if challenge_res == "pass":
            im = show_image(cam,question+" : ok")
            cv2.imshow('liveness_detection',im)
            if cv2.waitKey(1) &0xFF == ord('q'):
                break

            counter_ok_consecutives += 1
            if counter_ok_consecutives == limit_consecutives:
                counter_ok_questions += 1
                counter_try = 0
                counter_ok_consecutives = 0
                break
            else:
                continue

        elif challenge_res == "fail":
            counter_try += 1
            show_image(cam,question+" : fail")
        elif i_try == limit_try-1:
            break
            

    if counter_ok_questions ==  limit_questions:
        while True:
            im = show_image(cam,"LIFENESS SUCCESSFUL",color = (0,255,0))
            cv2.imshow('liveness_detection',im)
            if cv2.waitKey(1) &0xFF == ord('q'):
                break
    elif i_try == limit_try-1:
        while True:
            im = show_image(cam,"LIFENESS FAIL")
            cv2.imshow('liveness_detection',im)
            if cv2.waitKey(1) &0xFF == ord('q'):
                break
        break 

    else:
        continue
