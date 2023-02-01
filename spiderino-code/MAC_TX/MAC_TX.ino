#include <ESP8266WiFi.h>
#include <espnow.h>

#define RETRY_INTERVAL 5000
#define SEND_INTERVAL 10

char c;
String dataIn;
int8_t indexOfA, indexOfB, indexOfC, indexOfD;

String IR0, IR1, IR2, IR3;


// the following three settings must match the slave settings
uint8_t remoteMac[] = { 0x82, 0x88, 0x88, 0x88, 0x88, 0x88 };
const uint8_t channel = 1;
struct __attribute__((packed)) DataStruct {
  int IR0_TX;
  int IR1_TX;
  int IR2_TX;
  int IR3_TX;
};

DataStruct myData;

unsigned long sentStartTime;
unsigned long lastSentTime;

void sendData() {
  uint8_t bs[sizeof(myData)];
  memcpy(bs, &myData, sizeof(myData));

  sentStartTime = micros();
  esp_now_send(NULL, bs, sizeof(myData));  // NULL means send to all peers
}

void sendCallBackFunction(uint8_t* mac, uint8_t sendStatus) {
  unsigned long sentEndTime = micros();
  Serial.printf("Send To: %02x:%02x:%02x:%02x:%02x:%02x ", mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
  Serial.printf("IR0 : %d ", myData.IR0_TX);
  Serial.printf("IR1 : %d ", myData.IR1_TX);
  Serial.printf("IR2 : %d ", myData.IR2_TX);
  Serial.printf("IR3 : %d ", myData.IR3_TX);

  Serial.printf("Trip micros: %4lu, ", sentEndTime - sentStartTime);
  Serial.printf("Status: %s\n", (sendStatus == 0 ? "Success" : "Failed"));
}

void Parse_the_Data() {

  indexOfA = dataIn.indexOf("A");
  indexOfB = dataIn.indexOf("B");
  indexOfC = dataIn.indexOf("C");
  indexOfD = dataIn.indexOf("D");

  IR0 = dataIn.substring(0, indexOfA);
  IR1 = dataIn.substring(indexOfA + 1, indexOfB);
  IR2 = dataIn.substring(indexOfB + 1, indexOfC);
  IR3 = dataIn.substring(indexOfC + 1, indexOfD);
}

void setup() {
  WiFi.mode(WIFI_STA);  // Station mode for esp-now controller
  WiFi.disconnect();

  Serial.begin(9600);
  Serial.println();
  Serial.println("ESP-Now Transmitter");
  Serial.printf("Transmitter mac: %s \n", WiFi.macAddress().c_str());
  Serial.printf("Receiver mac: %02x:%02x:%02x:%02x:%02x:%02x\n", remoteMac[0], remoteMac[1], remoteMac[2], remoteMac[3], remoteMac[4], remoteMac[5]);
  Serial.printf("WiFi Channel: %i\n", channel);

  if (esp_now_init() != 0) {
    Serial.println("ESP_Now init failed...");
    delay(RETRY_INTERVAL);
    ESP.restart();
  }

  esp_now_set_self_role(ESP_NOW_ROLE_CONTROLLER);
  esp_now_add_peer(remoteMac, ESP_NOW_ROLE_SLAVE, channel, NULL, 0);
  esp_now_register_send_cb(sendCallBackFunction);
}

void loop() {

  if (millis() - lastSentTime >= SEND_INTERVAL) {
    lastSentTime += SEND_INTERVAL;

    while (Serial.available() > 0) {

      c = Serial.read();

      if (c != '\n'){
        dataIn += c;
      }

      if (c == '\n') {
        Parse_the_Data();
        c = 0;
        dataIn = "";
      }
    }

    myData.IR0_TX = IR0.toInt();
    myData.IR1_TX = IR1.toInt();
    myData.IR2_TX = IR2.toInt();
    myData.IR3_TX = IR3.toInt();

    sendData();
  }
}