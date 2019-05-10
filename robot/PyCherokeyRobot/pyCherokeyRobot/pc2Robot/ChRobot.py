
#from robot.PyCherokeyRobot.pyCherokeyRobot.CherokeyRobotInterface import *
import sys
sys.path.append('/home/mattapattu/DNF/robot/PyCherokeyRobot/pyCherokeyRobot')
from CherokeyRobotInterface import *

import rpyc

class ChRobot(CherokeyRobotInterface):

    def __init__(self, HOST = 'localhost', PORT = 5638):
        print("Waitting robot body connection...")
        self.c = rpyc.connect(HOST, PORT) 
        print("Body found!")

    def __del__(self):
        print("Byebye Robot!")
    
    def setRightSpeed(self, speed):
        self.c.root.setRightSpeed(speed)
    
    
    def setLeftSpeed(self, speed):
        self.c.root.setLeftSpeed(speed)

    def getPixyBlocks(self) :
        return self.c.root.getBlocks()
    

