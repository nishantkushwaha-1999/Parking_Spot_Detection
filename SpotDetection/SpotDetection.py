import sys
import os
path = os.getcwd()
parent = os.path.dirname(path)
sys.path.insert(0, parent)

import cv2
import numpy as np
from typing import Union
from ObjectDetection.ObjectDetection import ObjectDetection
from shapely.geometry import Polygon
from time import sleep

class SpotDetection():
    def __init__(self):
        self.OD = ObjectDetection(["car", "truck"])
        self.bboxes = None
        self.bbox_area = None
        self.spot_area = None
        self.frame = None
        
    def intersection(self, poly_x: list, poly_y: list, rect_x: list, rect_y: list):
        if len(poly_x)!=5:
            raise ValueError(f"Expected 5 x coordinates for polygon got {len(poly_x)}")
        elif len(poly_y)!=5:
            raise ValueError(f"Expected 5 y coordinates for polygon got {len(poly_y)}")
        elif len(rect_x)!=5:
            raise ValueError(f"Expected 5 x coordinates for rectangle got {len(rect_x)}")
        elif len(rect_y)!=5:
            raise ValueError(f"Expected 5 y coordinates for rectangle got {len(rect_y)}")
        else:
            bbox_cords = ((rect_x[0], rect_y[0]), (rect_x[1], rect_y[1]), (rect_x[2], rect_y[2]), 
                          (rect_x[3], rect_y[3]), (rect_x[4], rect_y[4]))
            bbox = Polygon(bbox_cords)
            bbox = Polygon(bbox_cords)
            if not bbox.is_valid:
                bbox = bbox.buffer(0)
            
            spot_cords = ((poly_x[0], poly_y[0]), (poly_x[1], poly_y[1]), (poly_x[2], poly_y[2]), 
                          (poly_x[3], poly_y[3]), (poly_x[4], poly_y[4]))
            spot = Polygon(spot_cords)
            if not spot.is_valid:
                spot = spot.buffer(0)
            
            intersection = bbox.intersection(spot).area
            return intersection
    
    def union(self, poly_x: list, poly_y: list, rect_x: list, rect_y: list):
        if len(poly_x)!=5:
            raise ValueError(f"Expected 5 x coordinates for polygon got {len(poly_x)}")
        elif len(poly_y)!=5:
            raise ValueError(f"Expected 5 y coordinates for polygon got {len(poly_y)}")
        elif len(rect_x)!=5:
            raise ValueError(f"Expected 5 x coordinates for rectangle got {len(rect_x)}")
        elif len(rect_y)!=5:
            raise ValueError(f"Expected 5 y coordinates for rectangle got {len(rect_y)}")
        else:
            bbox_cords = ((rect_x[0], rect_y[0]), (rect_x[1], rect_y[1]), (rect_x[2], rect_y[2]), 
                        (rect_x[3], rect_y[3]), (rect_x[4], rect_y[4]))
            bbox = Polygon(bbox_cords)
            if not bbox.is_valid:
                bbox = bbox.buffer(0)
            
            spot_cords = ((poly_x[0], poly_y[0]), (poly_x[1], poly_y[1]), (poly_x[2], poly_y[2]), 
                        (poly_x[3], poly_y[3]), (poly_x[4], poly_y[4]))
            spot = Polygon(spot_cords)
            if not spot.is_valid:
                spot = spot.buffer(0)
            
            union = spot.union(bbox).area
            return union
    
    def intersection_over_union(self, x_spot: list, y_spot: list, box_xs: list, box_ys: list):
        if len(x_spot)!=5:
            raise ValueError(f"Expected 5 x coordinates for polygon got {len(x_spot)}")
        elif len(y_spot)!=5:
            raise ValueError(f"Expected 5 y coordinates for polygon got {len(y_spot)}")
        elif len(box_xs)!=5:
            raise ValueError(f"Expected 5 x coordinates for rectangle got {len(box_xs)}")
        elif len(box_ys)!=5:
            raise ValueError(f"Expected 5 y coordinates for rectangle got {len(box_ys)}")
        else:
            if (max(x_spot)<min(box_xs)) or (min(x_spot)>max(box_xs)):
                intersection_area = 0
            elif (max(y_spot)<min(box_ys)) or (min(y_spot)>max(box_ys)):
                intersection_area = 0
            else:
                intersection_area = self.intersection(x_spot, y_spot, box_xs, box_ys)
        
            union_area = self.union(x_spot, y_spot, box_xs, box_ys)
            intersection_over_union = intersection_area / union_area
            return intersection_over_union
    
    def checkSpot(self, spots: dict):
        spot_status = {}
        for name, coords in spots.items():
            x_spot = []
            y_spot = []
            for item in coords:
                x_spot.append(item[0])
                y_spot.append(item[1])
            x_spot.append(x_spot[0])
            y_spot.append(y_spot[1])

            for box in self.bboxes:
                (x, y, x2, y2) = box
                box_xs = [x, x2, x2, x, x]
                box_ys = [y, y, y2, y2, y]

                iou = self.intersection_over_union(x_spot, y_spot, box_xs, box_ys)
                if iou < 0.20:
                    spot_status[name] = "Free"
                else:
                    spot_status[name] = "Occupied"
                    break
        return spot_status
    
    def getBboxes(self, frame):
        self.frame = frame
        self.bboxes = self.OD.getObjectBoxes(frame)
        return self.bboxes
    
    def highlightSpot(self, frame, spot_status, spots):
        for name, coords in spots.items():
            spot_coords = []
            for item in coords:
                spot_coords.append([item[0], item[1]])
            spot_coords.append(spot_coords[0])

            if spot_status[name] == 'Free':
                color = (0, 255, 0)
            elif spot_status[name] == "Occupied":
                color = (0, 0, 255)
            
            frame = cv2.polylines(self.frame, pts=[np.int32(spot_coords)], isClosed=False, color=color, thickness=2)
        return frame
