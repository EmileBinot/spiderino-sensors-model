#include <SoftwareSerial.h>

//----------------------MOTOR------------------------
#define SPEED 20
#define SPEED_ROTA 57
#define STOP 0
#define STOP_TIME 2000

int aPhase = 2;
int aEnable = 3;
int bPhase = 4;
int bEnable = 5;

int randomNumber, lastRandomNumber; 
int randomTime;

//----------------------ESP_DATA---------------------

SoftwareSerial toESP(7,6); //RX, TX 

int IRsensor_0 = 17;  
int IRsensor_1 = 16;
int IRsensor_2 = 15;
int IRsensor_3 = 14;

int IR0, IR1, IR2, IR3;

unsigned long start;

void setup() {

  //ESP_DATA
  toESP.begin(9600);
  Serial.begin(9600);
  
  pinMode(IRsensor_0, INPUT);
  pinMode(IRsensor_1, INPUT);
  pinMode(IRsensor_2, INPUT);  
  pinMode(IRsensor_3, INPUT);  

  //MOTOR
  pinMode(aPhase, OUTPUT);
  pinMode(aEnable, OUTPUT);
  pinMode(bPhase, OUTPUT);
  pinMode(bEnable, OUTPUT);

  digitalWrite(aPhase, LOW);
  digitalWrite(bPhase, LOW);
  analogWrite(aEnable, STOP);
  analogWrite(aEnable, STOP);
  
  delay(3000);

}

void loop() {

  randomNumber = random(1,5);
  randomTime = random(1,3)*1000;

  if(randomNumber == 1){
    walkForward();
    delay(randomTime);
  }

  else if(randomNumber == 2){
    walkBackward();
    delay(randomTime);
  }
  
  else if(randomNumber == 3 && lastRandomNumber != 4){
    turnLeft();
    delay(2000);
  }

  else if(randomNumber == 4 && lastRandomNumber != 3){
    turnRight();
    delay(2000);
  }

  walkStop();

  unsigned long now = millis();

  do{
  IR0 = analogRead(IRsensor_0);
  IR1 = analogRead(IRsensor_1);
  IR2 = analogRead(IRsensor_2);
  IR3 = analogRead(IRsensor_3);

  toESP.print(IR0); toESP.print("A");
  toESP.print(IR1); toESP.print("B");
  toESP.print(IR2); toESP.print("C");
  toESP.print(IR3); toESP.print("D");
  toESP.print("\n");

  now = millis();

  }while(now - start <= STOP_TIME);



  lastRandomNumber = randomNumber;

}



// Walk forward function
void walkForward() {

  digitalWrite(aPhase, HIGH);
  digitalWrite(bPhase, HIGH);
  analogWrite(aEnable, SPEED);
  analogWrite(bEnable, SPEED);
}

// Walk backward function
void walkBackward() {

  digitalWrite(aPhase, LOW);
  digitalWrite(bPhase, LOW);
  analogWrite(aEnable, SPEED);
  analogWrite(bEnable, SPEED);
}

// Stop walking function
void walkStop() {
  digitalWrite(aPhase, LOW);
  digitalWrite(bPhase, LOW);
  analogWrite(aEnable, STOP);
  analogWrite(bEnable, STOP);
  start = millis();
}

// Turn right function
void turnRight() {
  digitalWrite(aPhase, HIGH);
  digitalWrite(bPhase, HIGH);
  analogWrite(bEnable, STOP);
  analogWrite(aEnable, SPEED_ROTA);
}

// Turn left function
void turnLeft() {
  digitalWrite(aPhase, HIGH);
  digitalWrite(bPhase, HIGH);
  analogWrite(aEnable, STOP);
  analogWrite(bEnable, SPEED_ROTA);
}

