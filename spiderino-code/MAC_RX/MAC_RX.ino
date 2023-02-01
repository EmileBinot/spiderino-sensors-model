
#include <ESP8266WiFi.h>
#include <espnow.h>

#define RETRY_INTERVAL 5000

// the following 3 settings must match transmitter's settings
uint8_t mac[] = {0x82, 0x88, 0x88, 0x88, 0x88, 0x88};
const uint8_t channel = 1;
struct __attribute__((packed)) DataStruct {
  int IR0_TX;
  int IR1_TX;
  int IR2_TX;
  int IR3_TX;
};

DataStruct myData;

void receiveCallBackFunction(uint8_t *senderMac, uint8_t *incomingData, uint8_t len) {

  memcpy(&myData, incomingData, len);
  //Serial.printf("\r\nTransmitter MacAddr: %02x:%02x:%02x:%02x:%02x:%02x, ", senderMac[0], senderMac[1], senderMac[2], senderMac[3], senderMac[4], senderMac[5]);
  //Serial.printf("IR0 : %d, ", myData.IR0_TX);
  //Serial.printf("IR1 : %d\r\n", myData.IR1_TX);

  Serial.print(myData.IR0_TX);
  Serial.print(",");
  Serial.print(myData.IR1_TX);
  Serial.print(",");
  Serial.print(myData.IR2_TX);
  Serial.print(",");
  Serial.println(myData.IR3_TX);

}

void setup() {
  
  Serial.begin(9600);

  delay(1000);

  WiFi.mode(WIFI_AP);
  wifi_set_macaddr(SOFTAP_IF, &mac[0]);
  WiFi.disconnect();
  
  //Serial.println();
  //Serial.println("ESP-Now Receiver");
  //Serial.printf("Transmitter mac: %s\n", WiFi.macAddress().c_str());
  //Serial.printf("Receiver mac: %s\n", WiFi.softAPmacAddress().c_str());
  if (esp_now_init() != 0) {
    //Serial.println("ESP_Now init failed...");
    delay(RETRY_INTERVAL);
    ESP.restart();
  }
  esp_now_set_self_role(ESP_NOW_ROLE_SLAVE);
  esp_now_register_recv_cb(receiveCallBackFunction);
  //Serial.println("Slave ready. Waiting for messages...");

}

void loop() {


}
