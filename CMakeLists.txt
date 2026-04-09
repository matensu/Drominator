cmake_minimum_required(VERSION 3.16)
project(FPVDroneAI_Linux LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Qt MOC / RCC / UIC automatiques ──────────────────────────────────────────
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# ── Qt6 en priorité, fallback Qt5 ────────────────────────────────────────────
find_package(Qt6 COMPONENTS Core Widgets QUIET)
if (Qt6_FOUND)
    message(STATUS "Qt6 : ${Qt6_VERSION}")
    set(QT_LIBS Qt6::Core Qt6::Widgets)
else()
    find_package(Qt5 COMPONENTS Core Widgets REQUIRED)
    message(STATUS "Qt5 : ${Qt5_VERSION}")
    set(QT_LIBS Qt5::Core Qt5::Widgets)
endif()

# ── OpenCV ────────────────────────────────────────────────────────────────────
find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio dnn highgui)
message(STATUS "OpenCV : ${OpenCV_VERSION}")

# ── Threads ───────────────────────────────────────────────────────────────────
find_package(Threads REQUIRED)

# ── Exécutable ────────────────────────────────────────────────────────────────
add_executable(fpv_drone_ai
    fpv_drone_gui_linux.cpp
)

target_include_directories(fpv_drone_ai PRIVATE ${OpenCV_INCLUDE_DIRS})

target_link_libraries(fpv_drone_ai PRIVATE
    ${QT_LIBS}
    ${OpenCV_LIBS}
    Threads::Threads    # pthread sur Linux
)

# ── Flags de compilation optimisés ───────────────────────────────────────────
target_compile_options(fpv_drone_ai PRIVATE
    -O2
    -Wall
    -Wextra
)

# ─────────────────────────────────────────────────────────────────────────────
#  GUIDE INSTALLATION ET COMPILATION — Pop!_OS / Ubuntu 22.04+
# ─────────────────────────────────────────────────────────────────────────────
#
#  1. INSTALLER LES DÉPENDANCES (une seule fois)
#  ──────────────────────────────────────────────
#  sudo apt update
#  sudo apt install -y \
#      cmake build-essential \
#      qt6-base-dev libqt6widgets6 \
#      libopencv-dev \
#      libv4l-dev v4l-utils
#
#  Si Qt6 n'est pas disponible (Ubuntu 20.04) :
#  sudo apt install -y qtbase5-dev qt5-default
#
#  2. COMPILER
#  ──────────────────────────────────────────────
#  mkdir build && cd build
#  cmake ..
#  make -j$(nproc)
#
#  3. DONNER ACCÈS AU PORT SÉRIE (sans sudo)
#  ──────────────────────────────────────────
#  sudo usermod -aG dialout $USER
#  # puis se déconnecter / reconnecter (ou : newgrp dialout)
#
#  Vérifier que le module ELRS est détecté :
#  ls /dev/ttyUSB* /dev/ttyACM*
#  dmesg | tail -20   # voir le driver chargé
#
#  4. PRIORITÉS TEMPS RÉEL (optionnel mais recommandé)
#  ───────────────────────────────────────────────────
#  Option A — lancer en root :
#    sudo ./fpv_drone_ai
#
#  Option B — capabilities (recommandé, pas de root) :
#    sudo setcap cap_sys_nice+ep ./fpv_drone_ai
#    ./fpv_drone_ai
#
#  5. IDENTIFIER LA CAMÉRA DE CAPTURE FPV
#  ──────────────────────────────────────────
#  v4l2-ctl --list-devices
#  # Choisir l'index /dev/videoX correspondant à la carte de capture
#
#  6. TESTER L'UART SANS DRONE
#  ──────────────────────────────────────────
#  # Envoyer et recevoir via minicom :
#  sudo minicom -D /dev/ttyUSB0 -b 420000
#  # Ou avec cat (hex) :
#  cat /dev/ttyUSB0 | xxd | head
#
#  7. VÉRIFIER LES CHANNELS DANS BETAFLIGHT
#  ──────────────────────────────────────────
#  Ouvrir Betaflight Configurator > onglet Receiver
#  CH1-4 doivent bouger selon l'image caméra
#  CH5 = 1000 µs = drone DÉSARMÉ (sécurité garantie)
# ─────────────────────────────────────────────────────────────────────────────
