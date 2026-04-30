import pygame
import sys
import os

def tester_manette():
    # Initialisation de Pygame
    pygame.init()
    pygame.joystick.init()

    # Vérification de la présence d'une manette
    if pygame.joystick.get_count() == 0:
        print("❌ Erreur : Aucune manette détectée.")
        print("Vérifie que ta radio est allumée et en mode USB HID.")
        return

    # Sélection de la première manette
    manette = pygame.joystick.Joystick(0)
    manette.init()
    
    print(f"✅ Manette détectée : {manette.get_name()}")
    print("--- Test des directions (Appuie sur Ctrl+C pour quitter) ---")

    try:
        while True:
            # Actualise les événements de la manette
            pygame.event.pump()

            # Lecture des axes principaux (Ordre standard souvent 0, 1, 2, 3)
            # On arrondit pour éviter les micro-mouvements (Deadzone)
            roll = round(manette.get_axis(0), 2)   # Axe Horizontal Droite
            pitch = round(manette.get_axis(1), 2)  # Axe Vertical Droite
            
            # Nettoyage du terminal pour une lecture fluide
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"Analyse de la manette : {manette.get_name()}")
            print("-" * 40)

            # --- LOGIQUE DE DIRECTION ---
            # Seuil de 0.5 pour ne pas afficher de direction si le stick est au centre
            
            # Axe Horizontal (Roll / Aileron)
            if roll < -0.5:
                direction_h = "⬅️  GAUCHE"
            elif roll > 0.5:
                direction_h = "➡️  DROITE"
            else:
                direction_h = "⏺️  CENTRE (Horizontal)"

            # Axe Vertical (Pitch / Profondeur)
            # Note : Sur beaucoup de radios, le "Haut" est une valeur négative
            if pitch < -0.5:
                direction_v = "⬆️  HAUT"
            elif pitch > 0.5:
                direction_v = "⬇️  BAS"
            else:
                direction_v = "⏺️  CENTRE (Vertical)"

            # Affichage
            print(f"ROLL  (Axe 0) : {roll:5}  ->  {direction_h}")
            print(f"PITCH (Axe 1) : {pitch:5}  ->  {direction_v}")
            print("-" * 40)
            print("Bouge les sticks de droite pour tester !")

            pygame.time.wait(50) # Pause de 50ms pour ne pas surcharger le processeur

    except KeyboardInterrupt:
        print("\nTest terminé.")
        pygame.quit()

if __name__ == "__main__":
    tester_manette()