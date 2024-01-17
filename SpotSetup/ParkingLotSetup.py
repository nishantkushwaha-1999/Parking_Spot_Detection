import os
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.widgets as widgets

class ParkingSpotSetup():
    def __init__(self, videoPath, load=False, reset=False):
        self.co_ordinates = []
        self.image = None
        self.spot_co_ordinates = None
        self.spot_dict = {}
        self.video_path = videoPath
        self.key = None
        self.filename = "./SpotSetup/spots.json"
        self.imname = "./SpotSetup/parking_lot.jpg"

        if reset:
            self.reset()

        if not load:
            while self.key != "l":
                self.setup()
    
    def setup(self):
        self.videoCapture()

        self.fig, self.ax = plt.subplots()
        self.canvas = self.ax.figure.canvas
        self.ax.imshow(self.image)

        try:
            with open(self.filename, 'r') as json_file:
                data = json.load(json_file)
                for item in data.values():
                    self.co_ordinates.append(item)
            self.drawSpot(data)
        except FileNotFoundError:
            pass
        
        self.selectSpot()
        plt.connect("key_press_event", self.listner)
        plt.connect("key_press_event", self.saveAllSpots)
        plt.show()

    def videoCapture(self):
        if os.path.isfile(self.imname):
            self.image = cv2.imread(self.imname)
        else:
            path = self.video_path
            print("reading without file", path)
            video_capture = cv2.VideoCapture(path)
            cnt=0
            self.image = None
            while video_capture.isOpened():
                success, frame = video_capture.read()
                if not success:
                    break
                if cnt == 5:
                    self.image = frame[:, :, ::-1]
                    break
                cnt += 1
            video_capture.release()

    def reset(self):
        if os.path.isfile(self.imname):
            os.remove(self.imname)
        if os.path.isfile(self.filename):
            os.remove(self.filename)

    def drawSpot(self, spots):
        spots = spots.items()
        for spot in spots:
            name, co_ordinate = spot
            co_ordinate = np.array(co_ordinate)
            spot = patches.Polygon(co_ordinate, closed=True, alpha=0.65, facecolor='purple')
            self.ax.add_patch(spot)
    
    def selectSpot(self):
        self.spot_selector = widgets.PolygonSelector(self.ax, onselect=self.identifySpot, draw_bounding_box=True)

    def identifySpot(self, verts):
        self.spot_co_ordinates = []
        self.canvas.draw_idle()
        for i in range(len(verts)):
            self.spot_co_ordinates.append(list(verts[i]))
    
    def saveSpot(self):
        if len(self.spot_co_ordinates) == 4:
            print(self.spot_co_ordinates)
            self.co_ordinates.append(self.spot_co_ordinates)
            
            name = f"spot_{len(self.co_ordinates)+1}"
            self.spot_dict[name] = self.spot_co_ordinates
            try:
                with open(self.filename, 'r') as json_file:
                    data = json.load(json_file)
                    data.update(self.spot_dict)
                    data = self.renameSpots(data=data)
                with open(self.filename, 'w') as json_file:
                    json.dump(data, json_file)
            
            except FileNotFoundError:
                if not os.path.isfile(self.imname):
                    cv2.imwrite(self.imname, self.image)
                
                with open(self.filename, 'w') as json_file:
                    json.dump(self.spot_dict, json_file)
        
        elif len(self.spot_co_ordinates) != 4 and len(self.spot_co_ordinates) != 0:
            raise ValueError("Please select a Quadrilateral")
        
        elif len(self.spot_co_ordinates) == 0:
            raise ValueError("Encountered Empty array. Please Retry.")
    
    def listner(self, event):
        if event.key == 'p':
            self.key = "p"
            self.saveSpot()
            plt.close()
    
    def saveAllSpots(self, event):
        if event.key == "l":
            self.key = "l"
            self.spot_selector.disconnect_events()
            self.canvas.draw_idle()
            plt.close()
    
    def retrieveSpots(self):
        if not os.path.isfile(self.filename):
            raise FileNotFoundError("Parking Spot file not detected")
        
        with open(self.filename, 'r') as json_file:
            data = json.load(json_file)
            return data
    
    def renameSpots(self, data):
        renamed = {}
        for i, coord in enumerate(data.values()):
            renamed[f"spot_{i+1}"] = coord
        return renamed