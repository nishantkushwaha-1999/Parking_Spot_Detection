import sys
import os
path = os.getcwd()
parent = os.path.dirname(path)
sys.path.insert(0, parent)

# import time
import cv2
from SpotSetup.ParkingLotSetup import ParkingSpotSetup as PSS
from SpotDetection.SpotDetection import SpotDetection

if __name__=="__main__":
    path = input("Provide the video path...")

    if os.path.isfile("SpotSetup/parking_lot.jpg"):
        while True:
            detect = input("Detected old setup file. Continue with previous setup?. [Y/n]")
            if detect == "Y":
                spot_selector = PSS(path, load=True)
                break
            elif detect == "n":
                detect = input("Do you want to modify previous setup?. Denying will start a new setup. [Y/n]")
                if detect == "n":
                    spot_selector = PSS(videoPath=path, reset=True)
                    break
                elif detect == "Y":
                    spot_selector = PSS(videoPath=path)
                    break
                else:
                    print("Enter a Valid Input.")
            else:
                print("Enter a Valid Input.")
    else:
        spot_selector = PSS(videoPath=path, reset=True)

    spots = spot_selector.retrieveSpots()
    spot_detection = SpotDetection()

    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_FPS, 5)
    while True:
        # start = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        boxes = spot_detection.getBboxes(frame=frame)
        spot_status = spot_detection.checkSpot(spots=spots)
        frame = spot_detection.highlightSpot(frame=frame, spot_status=spot_status, spots=spots)
        cv2.imshow("Img", frame)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        # break
    # end = time.time()
    # print("The time of execution of above program is :",
    #       (end-start), "s")
    
    cap.release()
    cv2.destroyAllWindows()