import pygame
import serial
import time

# --- CONFIGURATION ---
SERIAL_PORT = 'COM3'  # À adapter (ex: /dev/ttyACM0 sur Linux)
BAUD_RATE = 420000    # Standard pour l'ExpressLRS / CRSF
REFRESH_RATE = 0.02   # 50Hz (20ms)

class DroneAIPont:
    def __init__(self):
        # Initialisation de la manette (USB HID)
        pygame.init()
        pygame.joystick.init()
        
        if pygame.joystick.get_count() == 0:
            raise Exception("Radiocommande non détectée. Vérifiez le mode USB HID.")
        
        self.controller = pygame.joystick.Joystick(0)
        self.controller.init()
        print(f"Connecté à : {self.controller.get_name()}")

        # Initialisation de la liaison vers le module ELRS
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0)
        except:
            print("Attention : Port série non trouvé. Mode simulation actif.")
            self.ser = None

    def get_manual_sticks(self):
        """Récupère les axes de la radio (-1.0 à 1.0)"""
        pygame.event.pump()
        return {
            "roll":  self.controller.get_axis(0),
            "pitch": self.controller.get_axis(1),
            "throttle": self.controller.get_axis(2),
            "yaw":   self.controller.get_axis(3),
            "arm_switch": self.controller.get_button(0) # Bouton d'armement
        }

    def mix_ai_commands(self, manual, ai_cam_data):
        """
        Fusionne les données manuelles et l'IA.
        Exemple : L'IA caméra ajuste le Pitch pour garder la cible au centre.
        """
        final_commands = manual.copy()
        
        # Logique d'assistance IA : 
        # Si l'IA détecte quelque chose, elle modifie légèrement le Pitch/Yaw
        if ai_cam_data['target_locked']:
            final_commands['pitch'] += ai_cam_data['correction_y'] * 0.5
            final_commands['yaw']   += ai_cam_data['correction_x'] * 0.5
            
        # On s'assure de rester dans les limites [-1, 1]
        for key in ['roll', 'pitch', 'throttle', 'yaw']:
            final_commands[key] = max(-1.0, min(1.0, final_commands[key]))
            
        return final_commands

    def send_crsf(self, commands):
        """Convertit et envoie au protocole CRSF"""
        # Conversion -1/1 vers 172/1811 (Plage standard CRSF/ELRS)
        channels = [992] * 16
        channels[0] = int((commands['roll'] + 1) * 819.5 + 172)
        channels[1] = int((commands['pitch'] + 1) * 819.5 + 172)
        channels[2] = int((commands['throttle'] + 1) * 819.5 + 172)
        channels[3] = int((commands['yaw'] + 1) * 819.5 + 172)
        
        # Ici on injecterait la fonction de packaging CRSF (voir réponse précédente)
        if self.ser:
            # self.ser.write(self.pack_crsf(channels))
            pass

    def run(self):
        print("Système démarré. Appuyez sur Ctrl+C pour stopper.")
        try:
            while True:
                # 1. Lecture Radio
                manual = self.get_manual_sticks()
                
                # 2. Simulation de l'entrée de ton IA Caméra existante
                # ai_cam_data = ton_ia_camera.get_inference()
                ai_cam_data = {'target_locked': False, 'correction_x': 0, 'correction_y': 0}
                
                # 3. Fusion et Envoi
                final = self.mix_ai_commands(manual, ai_cam_data)
                self.send_crsf(final)
                
                time.sleep(REFRESH_RATE)
        except KeyboardInterrupt:
            print("Arrêt de sécurité.")

if __name__ == "__main__":
    bridge = DroneAIPont()
    bridge.run()