from pyCherokeyRobot.pc2Robot.ChRobot import *

import time

def main():
    robot = ChRobot(HOST="raspberrypi.local")

    robot.setRightSpeed(0.1)
    robot.setLeftSpeed(0.1)
    
    time.sleep(2)
    
    robot.setRightSpeed(0.99)
    robot.setLeftSpeed(0.99)
    
    time.sleep(2)

    robot.setRightSpeed(0.5)
    robot.setLeftSpeed(0.5)
    
    
    quit()
   
    
        
if __name__ == "__main__":
    main()
