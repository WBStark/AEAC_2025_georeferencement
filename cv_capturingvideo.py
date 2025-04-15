import cv2
import time


           ### UNDERSTAND HOW TO GET A PHOTO FROM THE CAM USING NYMPY ARRAY or matplotlib idk
stream = cv2.VideoCapture(1)

if not stream.isOpened():
    print("no stream")
    exit()


i = 0
while(True):

    ret, frame = stream.read()
    if not ret:
        print("no more stream")
        break
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow("webcam", frame)
    #time.sleep(.5)
    if cv2.waitKey(1) == ord('-'):
        cv2.imwrite("./CL_CPU_"+str(i)+".png", frame)
        i = i + 1
    if cv2.waitKey(1) == ord('q'):
            break
stream.release()
cv2.destroyAllWindows()





#BGR
#img = cv2.imread("cat.jfif", cv2.IMREAD_COLOR)
#print(img.size)
#print(img.shape)
#for i in range(img.shape[0]):
#   for j in range(img.shape[1]):
#       img[i, j] = max(254, img[i, j] * 2)
#rbg_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#print(rbg_image[0, 0])
#cv2.imshow("cat",img)
#cv2.waitKey(0)


#gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imwrite("gray_cat.jpg", gray_img)
#gray_img2 = cv2.imread("gray_cat.jpg", cv2.IMREAD_COLOR)
#cv2.imshow("kitty", gray_img2)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

