from pyCherokeyRobot.CherokeyRobotInterface import *

import rpyc

import serial


class InRobotOrderProcessor(CherokeyRobotInterface, rpyc.Service):


    def __init__(self, serialPort = "/dev/ttyACM0"):
        print("- Robot Order Processor -")
        self.comPort = serial.Serial(serialPort, 115200, timeout=0, parity=serial.PARITY_ODD, rtscts=2)
    
    def on_connect(self, conn):
        print("Brain connected!")
        pass

    def on_disconnect(self, conn):
        print("Byebye brain!")
        pass

    def __setSpeed(self, side, speed): 
        self.comPort.write((":{}{}:\n".format(side, int(512 * speed - 256))).encode(encoding='utf-8', errors='strict')) 
        self.comPort.flush()


    def setRightSpeed(self, speed):
        #print("Set right speed to {}".format(speed))
        self.__setSpeed('r', speed)
    
    
    def setLeftSpeed(self, speed):
        #print("Set left speed to {}".format(speed))
        self.__setSpeed('l', speed)
        
        
    



