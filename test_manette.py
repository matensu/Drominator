import pygame
import os

def tester_manette_complete():
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("❌ Erreur : Aucune manette détectée.")
        return

    manette = pygame.joystick.Joystick(0)
    manette.init()
    
    print(f"✅ Analyse de : {manette.get_name()}")

    try:
        while True:
            pygame.event.pump()

            # --- LECTURE DES AXES ---
            # Stick Droit (Direction)
            roll = round(manette.get_axis(0), 2)   # Horizontal
            pitch = round(manette.get_axis(1), 2)  # Vertical
            
            # Stick Gauche (Puissance / Gaz)
            # Souvent l'axe 2 ou 3 selon la radio
            throttle = round(manette.get_axis(2), 2) 

            # Nettoyage console
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"--- TEST MANETTE FPV ---")
            print(f"Modèle : {manette.get_name()}")
            print("-" * 30)

            # --- LOGIQUE PUISSANCE MOTEUR (THROTTLE) ---
            if throttle <= -0.95:
                statut_moteur = "🛑 ARRÊT (Puissance : -1)"
            elif -0.05 <= throttle <= 0.05:
                statut_moteur = "⚖️  STABLE (Puissance : 0)"
            elif throttle >= 0.95:
                statut_moteur = "🚀 PLEINE PUISSANCE (Puissance : 1)"
            else:
                statut_moteur = f"⚙️  En mouvement ({throttle})"

            # --- LOGIQUE DIRECTION (PITCH / ROLL) ---
            dir_h = "CENTRE"
            if roll < -0.5: dir_h = "GAUCHE ⬅️"
            if roll > 0.5:  dir_h = "DROITE ➡️"

            dir_v = "CENTRE"
            if pitch < -0.5: dir_v = "HAUT ⬆️"
            if pitch > 0.5:  dir_v = "BAS ⬇️"

            # --- AFFICHAGE FINAL ---
            print(f"MOTEURS  [Axe 2] : {throttle:5} -> {statut_moteur}")
            print(f"ROLL     [Axe 0] : {roll:5} -> {dir_h}")
            print(f"PITCH    [Axe 1] : {pitch:5} -> {dir_v}")
            print("-" * 30)
            print("Utilise le stick de gauche pour la puissance")
            print("Utilise le stick de droite pour la direction")

            pygame.time.wait(100)

    except KeyboardInterrupt:
        print("\nArrêt du test.")
        pygame.quit()

if __name__ == "__main__":
    tester_manette_complete()