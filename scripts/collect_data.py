import csv
import serial
import time
from pathlib import Path

PORT = "COM3"
BAUD_RATE = 115200
DURATION = 15 * 60

output_dir = Path("data/raw")
output_dir.mkdir(parents=True, exist_ok=True)

filename = output_dir / f"normal_01.csv"

ser = serial.Serial(PORT, BAUD_RATE, timeout=1)

time.sleep(2)

print("Collecting sensor data...")
print(f"Saving to: {filename}")
print("Press Ctrl+C to stop.")

with open(filename, "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "timestamp_ms",
        "temperature_C",
        "accel_x",
        "accel_y",
        "accel_z",
        "rpm"
    ])

    try:
        while True:
            line = ser.readline().decode(
                "utf-8",
                errors="ignore"
            ).strip()

            if not line:
                continue

            if line.startswith("timestamp_ms"):
                continue

            values = line.split(",")

            if len(values) != 6:
                continue

            try:
                values = [
                    float(value)
                    for value in values
                ]
            except ValueError:
                continue

            writer.writerow(values)
            file.flush()

            print(values)

    except KeyboardInterrupt:
        print("\nData collection stopped.")

ser.close()

print(f"Dataset saved: {filename}")