#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::VideoCapture cap;
    
    // Le port de votre récepteur (4 ou 5 selon ce qu'on a vu)
    int camera_index = 4; 

    std::cout << "[INIT] Tentative d'ouverture de /dev/video" << camera_index << "..." << std::endl;

    // Ouverture forcée via V4L2
    cap.open(camera_index, cv::CAP_V4L2);
    if (!cap.isOpened()) {
        std::cerr << "[ERREUR] Impossible d'ouvrir la camera !" << std::endl;
        return -1;
    }

    // ==== CONFIGURATION MAGIQUE POUR MACROSILICON EWRF ====
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 720);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FPS, 60);
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    // ======================================================

    std::cout << "[OK] Flux video actif ! Appuyez sur la touche 'Echap' de votre clavier pour quitter." << std::endl;

    cv::Mat frame;
    
    // Boucle infinie d'affichage
    while (true) {
        cap >> frame; // Capture une nouvelle image
        
        if (frame.empty()) {
            std::cerr << "[ATTENTION] Image vide recue..." << std::endl;
            continue; // On ignore et on passe à la suivante
        }

        // Affiche l'image dans une fenêtre
        cv::imshow("RETOUR VIDEO FPV", frame);

        // Attend 1 milliseconde et vérifie si on a appuyé sur Echap (code 27)
        if (cv::waitKey(1) == 27) {
            break;
        }
    }

    // Nettoyage à la fermeture
    cap.release();
    cv::destroyAllWindows();
    std::cout << "[FIN] Camera fermee proprement." << std::endl;
    
    return 0;
}