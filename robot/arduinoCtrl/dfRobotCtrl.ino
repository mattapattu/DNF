// Controlling a servo position using a potentiometer (variable resistor) 
// by Michal Rinott <http://people.interaction-ivrea.it/m.rinott> 

#include <Servo.h> 
 
Servo headServo;  // create servo object to control a servo 

int LEFT = 0;
int RIGHT = 1;

int m_en_ports[2];
int m_pwm_ports[2];

char dir;
String val = "";

boolean readingValue;

int parseWheel(String msg){
  if(msg.charAt(0) == 'l')
    return LEFT; 
  else
    return RIGHT;  
}

int parseSpeed(String msg){
   return msg.substring(1).toInt(); 
}

void wheelsRotationSpeed(int wheel, int speed){
  Serial.print(wheel);
  Serial.println(speed);
  if(speed > 0){
    digitalWrite(m_en_ports[wheel], HIGH);
  }else{
    digitalWrite(m_en_ports[wheel], LOW);      
  }
  analogWrite(m_pwm_ports[wheel], abs(speed));
}
  
void headGotoX(int val){
  val = map(val, 0, 1023, 0, 179);     // scale it to use it with the servo (value between 0 and 180) 
  headServo.write(val);                  // sets the servo position according to the scaled value 
}


void setup() 
{ 
  //headServo.attach(9);  // attaches the servo on pin 9 to the servo object 
    
  m_en_ports[LEFT] = 4;
  m_en_ports[RIGHT] = 7;
  m_pwm_ports[LEFT] = 5;
  m_pwm_ports[RIGHT] = 6;
  
  Serial.begin(115200, SERIAL_8N2);
  
  readingValue = false;
} 
 
void loop() 
{   
  
} 

/*
  SerialEvent occurs whenever a new data comes in the hardware serial RX. This
  routine is run between each time loop() runs, so using delay inside loop can
  delay response. Multiple bytes of data may be available.
*/
void serialEvent() {
  while (Serial.available()) {
    // get the new byte:
    char inChar = (char)Serial.read();
    // add it to the inputString:
    // if the incoming character is a newline, set a flag so the main loop can
    // do something about it:
    if (inChar == ':') {
      readingValue = !readingValue;
      
      if(readingValue){
        val = "";    
      }else{
        //Serial.println(val);  
        wheelsRotationSpeed(parseWheel(val), parseSpeed(val));
      }
    }else{
       if(readingValue){
         val += inChar;
       } 
    }
  }
}

