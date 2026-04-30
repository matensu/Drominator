import pygame
import serial
import time

# --- CONFIGURATION ---
SERIAL_PORT = 'COM3'  # À CHANGER selon ton PC
BAUD_RATE = 420000    # Vitesse ELRS
CH_ORDER = [0, 1, 2, 3] # Ordre standard (Roll, Pitch, Throttle, Yaw)

class ELRSTestBridge:
    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        
        if pygame.joystick.get_count() == 0:
            print("ERREUR : Manette non détectée. Est-elle en mode USB HID ?")
            exit()
            
        self.controller = pygame.joystick.Joystick(0)
        self.controller.init()
        
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0)
            print(f"Connecté à {self.controller.get_name()} | Port: {SERIAL_PORT}")
        except Exception as e:
            print(f"ERREUR Port Série : {e}")
            self.ser = None

    def crsf_crc8(self, data):
        crc = 0
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x80: crc = (crc << 1) ^ 0xD5
                else: crc <<= 1
                crc &= 0xFF
        return crc

    def send_test_frame(self):
        pygame.event.pump()
        
        # 1. Lecture des 4 axes principaux
        # On convertit le -1/+1 en 172/1811 (Format CRSF)
        channels = [992] * 16
        for i in range(4):
            val = self.controller.get_axis(CH_ORDER[i])
            channels[i] = int((val + 1) * 819.5 + 172)

        # 2. Affichage Debug (pour vérifier que les chiffres bougent)
        print(f"CH1:{channels[0]} | CH2:{channels[1]} | CH3:{channels[2]} | CH4:{channels[3]}", end='\r')

        # 3. Encodage CRSF
        payload = bytearray()
        bits = 0
        bit_count = 0
        for ch in channels:
            bits |= (ch & 0x07FF) << bit_count
            bit_count += 11
            while bit_count >= 8:
                payload.append(bits & 0xFF)
                bits >>= 8
                bit_count -= 8
        
        frame = bytearray([0xEE, 24, 0x16]) + payload
        frame.append(self.crsf_crc8(frame[2:]))
        
        if self.ser:
            self.ser.write(frame)

    def start(self):
        print("Test en cours... Bouge les sticks ! (Ctrl+C pour arrêter)")
        try:
            while True:
                self.send_test_frame()
                time.sleep(0.01) # 100Hz
        except KeyboardInterrupt:
            print("\nTest terminé.")
            if self.ser: self.ser.close()

if __name__ == "__main__":
    tester = ELRSTestBridge()
    tester.start()