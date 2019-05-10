from pyCherokeyRobot.pc2Robot.ChRobot import *

import time

def main():
    robot = ChRobot(HOST="192.168.137.254")

    robot.setRightSpeed(0.40)
    robot.setLeftSpeed(0.30)

    time.sleep(2)

    #robot.setRightSpeed(0.45)
    #robot.setLeftSpeed(0.27)
    
    time.sleep(5)
    

    robot.setRightSpeed(0.5)
    robot.setLeftSpeed(0.5)
    
    
    quit()
   
    
        
if __name__ == "__main__":
    main()
