from pyCherokeyRobot.robot2Pc.inRobotOrderProcessor import *
from rpyc.utils.server import ThreadedServer

def main():
    t = ThreadedServer(InRobotOrderProcessor(), port=5638, protocol_config={
    'allow_public_attrs': True,})
    t.start()
    
    quit()
   
    
        
if __name__ == "__main__":
    main()
