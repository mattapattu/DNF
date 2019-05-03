from abc import abstractmethod


class CherokeyRobotInterface:

    @abstractmethod
    def setRightSpeed(self, speed): pass
    
    @abstractmethod
    def setLeftSpeed(self, speed): pass

    
