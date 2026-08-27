#include <Arduino.h>
#include <Wire.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_ADXL345_U.h>

#define TEMP_PIN 4
#define HALL_PIN 5
#define SDA_PIN 21
#define SCL_PIN 22

OneWire oneWire(TEMP_PIN);
DallasTemperature tempSensor(&oneWire);
Adafruit_ADXL345_Unified adxl(12345);

volatile uint32_t pulseCount = 0;

bool adxlOK = false;

float temperature = NAN;
float rpm = 0.0;

uint32_t lastTempTime = 0;
uint32_t lastRpmTime = 0;
uint32_t lastPulseCount = 0;

void IRAM_ATTR hallISR()
{
    pulseCount++;
}

void setup()
{
    Serial.begin(115200);
    delay(1000);

    Wire.begin(SDA_PIN, SCL_PIN);

    tempSensor.begin();

    pinMode(HALL_PIN, INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(HALL_PIN), hallISR, FALLING);

    if (adxl.begin())
    {
        adxlOK = true;
        adxl.setRange(ADXL345_RANGE_16_G);
        adxl.setDataRate(ADXL345_DATARATE_100_HZ);
    }

    Serial.println("timestamp_ms,temperature_C,accel_x,accel_y,accel_z,rpm");
}

void loop()
{
    uint32_t now = millis();

    if (now - lastTempTime >= 1000)
    {
        lastTempTime = now;

        tempSensor.requestTemperatures();

        float value = tempSensor.getTempCByIndex(0);

        if (value != DEVICE_DISCONNECTED_C)
        {
            temperature = value;
        }
    }

    if (now - lastRpmTime >= 1000)
    {
        noInterrupts();
        uint32_t currentPulses = pulseCount;
        interrupts();

        uint32_t pulses = currentPulses - lastPulseCount;

        rpm = pulses * 60.0;

        lastPulseCount = currentPulses;
        lastRpmTime = now;
    }

    float ax = NAN;
    float ay = NAN;
    float az = NAN;

    if (adxlOK)
    {
        sensors_event_t event;
        adxl.getEvent(&event);

        ax = event.acceleration.x;
        ay = event.acceleration.y;
        az = event.acceleration.z;
    }

    Serial.print(now);
    Serial.print(",");

    if (isnan(temperature))
        Serial.print("NaN");
    else
        Serial.print(temperature, 2);

    Serial.print(",");

    if (isnan(ax))
        Serial.print("NaN");
    else
        Serial.print(ax, 4);

    Serial.print(",");

    if (isnan(ay))
        Serial.print("NaN");
    else
        Serial.print(ay, 4);

    Serial.print(",");

    if (isnan(az))
        Serial.print("NaN");
    else
        Serial.print(az, 4);

    Serial.print(",");
    Serial.println(rpm, 2);

    delay(10);
}