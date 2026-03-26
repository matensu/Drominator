/**
 * ============================================================
 *  FPV DRONE AI CONTROLLER — ExpressLRS / CRSF over UART
 *  Auteur   : Expert Robotique C++
 *  Cible    : Windows 10/11 (MinGW-w64 / MSVC)
 *  Dépend.  : OpenCV 4.x (précompilé Windows)
 *  Lier     : -lwinmm  (timeBeginPeriod)
 * ============================================================
 *
 *  PIPELINE :
 *    [CaptureThread] → CircularBuffer<Frame>
 *                              ↓
 *    [AIThread]      → CircularBuffer<FlightCommand>
 *                              ↓
 *    [CRSFThread]    → COMx (Win32 API) → Module TX ELRS
 *                                                ↓
 *                                           Drone FPV
 *
 *  SÉCURITÉ :
 *    - Démarrer SANS hélices lors des premiers tests
 *    - FAILSAFE : throttle → 0 si aucune commande IA depuis 500ms
 *    - Arming switch CH5 = 1000 µs par défaut (désarmé)
 *    - Toutes les valeurs RC sont clampées [1000..2000] µs
 * ============================================================
 */

// ── Windows headers — TOUJOURS EN PREMIER ─────────────────────────────────────
#ifndef NOMINMAX
#  define NOMINMAX           // évite que windows.h écrase std::min/max
#endif
#ifndef WIN32_LEAN_AND_MEAN
#  define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>         // HANDLE, CreateFile, SetCommState, WriteFile…
#include <timeapi.h>         // timeBeginPeriod / timeEndPeriod

// ── Standard C++ ─────────────────────────────────────────────────────────────
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// ── OpenCV ────────────────────────────────────────────────────────────────────
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

// ─────────────────────────────────────────────────────────────────────────────
//  CONFIGURATION GLOBALE
// ─────────────────────────────────────────────────────────────────────────────
namespace Config {
    // Vidéo
    constexpr int    CAM_INDEX       = 0;            // 0 = première caméra DirectShow
    constexpr int    CAM_WIDTH       = 640;
    constexpr int    CAM_HEIGHT      = 480;
    constexpr int    CAM_FPS         = 60;

    // UART / CRSF  — adapter le numéro de port COM
    constexpr const char* UART_DEV   = "\\\\.\\COM3"; // ex. "\\\\.\\COM10"
    constexpr DWORD  UART_BAUD       = 420000;         // 420 kbaud standard CRSF
    constexpr int    CRSF_FREQ_HZ    = 250;            // Hz d'envoi

    // IA
    constexpr int    AI_MAX_MS       = 30;             // budget temps par frame
    constexpr float  OBSTACLE_THRESH = 0.35f;          // seuil densité de bords

    // RC : valeurs en µs [1000..2000], neutre = 1500
    constexpr uint16_t RC_MID        = 1500;
    constexpr uint16_t RC_MIN        = 1000;
    constexpr uint16_t RC_MAX        = 2000;
    constexpr uint16_t RC_DISARM     = 1000;

    // Tailles des buffers circulaires (doivent être des puissances de 2)
    constexpr size_t FRAME_BUF_SIZE  = 4;
    constexpr size_t CMD_BUF_SIZE    = 8;

    // YOLO ONNX — laisser vide pour utiliser la détection de contours
    constexpr const char* YOLO_MODEL = ""; // ex. "yolov8n.onnx"
    constexpr float YOLO_CONF        = 0.5f;
}

// ─────────────────────────────────────────────────────────────────────────────
//  STRUCTURES
// ─────────────────────────────────────────────────────────────────────────────
using RCChannels = std::array<uint16_t, 16>;

struct FlightCommand {
    float roll     = 0.f;   // -1 (gauche)  .. +1 (droite)
    float pitch    = 0.f;   // -1 (avant)   .. +1 (arrière)
    float yaw      = 0.f;   // -1 (CCW)     .. +1 (CW)
    float throttle = 0.f;   //  0 (min)     .. +1 (max)
    bool  obstacle = false;
};

struct VideoFrame {
    cv::Mat img;
    std::chrono::steady_clock::time_point ts;
};

// ─────────────────────────────────────────────────────────────────────────────
//  BUFFER CIRCULAIRE LOCK-FREE (SPSC — Single Producer / Single Consumer)
// ─────────────────────────────────────────────────────────────────────────────
template<typename T, size_t N>
class CircularBuffer {
    static_assert((N & (N - 1)) == 0, "N doit etre une puissance de 2");
public:
    bool push(T&& item) {
        const size_t w    = write_.load(std::memory_order_relaxed);
        const size_t next = (w + 1) & (N - 1);
        if (next == read_.load(std::memory_order_acquire)) return false; // plein
        buf_[w] = std::move(item);
        write_.store(next, std::memory_order_release);
        return true;
    }

    bool pop(T& item) {
        const size_t r = read_.load(std::memory_order_relaxed);
        if (r == write_.load(std::memory_order_acquire)) return false; // vide
        item = std::move(buf_[r]);
        read_.store((r + 1) & (N - 1), std::memory_order_release);
        return true;
    }

    // Vide le buffer et retourne uniquement l'élément le plus récent
    bool popLatest(T& item) {
        bool got = false;
        T tmp;
        while (pop(tmp)) { item = std::move(tmp); got = true; }
        return got;
    }

private:
    std::array<T, N>    buf_{};
    std::atomic<size_t> write_{0};
    std::atomic<size_t> read_{0};
};

// ─────────────────────────────────────────────────────────────────────────────
//  CRSF — Encodage et CRC-8/DVB-S2
//  Référence : https://github.com/crsf-wg/crsf/wiki
// ─────────────────────────────────────────────────────────────────────────────
namespace CRSF {
    constexpr uint8_t SYNC_BYTE        = 0xC8;
    constexpr uint8_t TYPE_RC_CHANNELS = 0x16;
    constexpr size_t  FRAME_LEN        = 26; // sync(1)+len(1)+type(1)+payload(22)+crc(1)

    // Polynôme DVB-S2 : 0xD5
    static uint8_t crc8(const uint8_t* data, size_t len) {
        uint8_t crc = 0;
        for (size_t i = 0; i < len; i++) {
            crc ^= data[i];
            for (int b = 0; b < 8; b++)
                crc = (crc & 0x80) ? (uint8_t)((crc << 1) ^ 0xD5)
                                   : (uint8_t)(crc << 1);
        }
        return crc;
    }

    /**
     * packCRSF() — encode 16 canaux 11 bits dans 22 octets payload CRSF
     *   Conversion µs → raw : raw = (µs - 1000) * 1639 / 1000 + 172
     *   Plage raw : 172 (1000µs) .. 1811 (2000µs), centre = 992
     */
    std::array<uint8_t, FRAME_LEN> packCRSF(const RCChannels& channels) {
        auto toRaw = [](uint16_t us) -> uint16_t {
            if (us < 1000) us = 1000;
            if (us > 2000) us = 2000;
            return (uint16_t)((us - 1000u) * 1639u / 1000u + 172u);
        };

        uint16_t raw[16];
        for (int i = 0; i < 16; i++) raw[i] = toRaw(channels[i]);

        // Pack 16 × 11 bits → 22 octets (little-endian)
        uint8_t payload[22] = {};
        int bitPos = 0;
        for (int ch = 0; ch < 16; ch++) {
            for (int b = 0; b < 11; b++) {
                if (raw[ch] & (1 << b))
                    payload[bitPos / 8] |= (uint8_t)(1u << (bitPos % 8));
                bitPos++;
            }
        }

        std::array<uint8_t, FRAME_LEN> frame{};
        frame[0] = SYNC_BYTE;
        frame[1] = 24;                // len = type(1) + payload(22) + crc(1)
        frame[2] = TYPE_RC_CHANNELS;
        std::memcpy(&frame[3], payload, 22);
        frame[25] = crc8(&frame[2], 23); // CRC sur TYPE + PAYLOAD
        return frame;
    }
} // namespace CRSF

// ─────────────────────────────────────────────────────────────────────────────
//  UART — Win32 API (CreateFile / SetCommState / WriteFile)
//  Remplace entièrement termios / open / write de Linux
// ─────────────────────────────────────────────────────────────────────────────
class UARTPort {
public:
    explicit UARTPort(const std::string& portName, DWORD baud) {
        // Syntaxe "\\\\.\\COMx" obligatoire pour COM > 9
        handle_ = CreateFileA(
            portName.c_str(),
            GENERIC_READ | GENERIC_WRITE,
            0,          // pas de partage
            nullptr,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL, // mode synchrone (pas d'overlapped)
            nullptr
        );
        if (handle_ == INVALID_HANDLE_VALUE) {
            throw std::runtime_error(
                "Impossible d'ouvrir " + portName +
                " (erreur Win32 #" + std::to_string(GetLastError()) + ")\n"
                "-> Verifier le port dans le Gestionnaire de peripheriques");
        }

        // Configurer le port série
        DCB dcb{};
        dcb.DCBlength = sizeof(DCB);
        if (!GetCommState(handle_, &dcb))
            throw std::runtime_error("GetCommState echoue");

        dcb.BaudRate = baud;
        dcb.ByteSize = 8;
        dcb.Parity   = NOPARITY;
        dcb.StopBits = ONESTOPBIT;
        dcb.fBinary  = TRUE;
        dcb.fParity  = FALSE;
        dcb.fOutxCtsFlow = FALSE;
        dcb.fOutxDsrFlow = FALSE;
        dcb.fDtrControl  = DTR_CONTROL_DISABLE;
        dcb.fRtsControl  = RTS_CONTROL_DISABLE;
        dcb.fOutX = FALSE;
        dcb.fInX  = FALSE;

        if (!SetCommState(handle_, &dcb))
            throw std::runtime_error(
                "SetCommState echoue — baud " + std::to_string(baud) +
                " supporte par ce driver ?");

        // Timeouts : lecture non bloquante, écriture avec timeout 50 ms
        COMMTIMEOUTS to{};
        to.ReadIntervalTimeout         = MAXDWORD;
        to.ReadTotalTimeoutConstant    = 0;
        to.ReadTotalTimeoutMultiplier  = 0;
        to.WriteTotalTimeoutConstant   = 50;
        to.WriteTotalTimeoutMultiplier = 0;
        SetCommTimeouts(handle_, &to);

        PurgeComm(handle_, PURGE_RXCLEAR | PURGE_TXCLEAR);
        std::cout << "[UART] " << portName << " ouvert @ " << baud << " baud\n";
    }

    ~UARTPort() {
        if (handle_ != INVALID_HANDLE_VALUE) CloseHandle(handle_);
    }

    UARTPort(const UARTPort&)            = delete;
    UARTPort& operator=(const UARTPort&) = delete;

    /**
     * sendCRSF() — envoie une frame CRSF (26 octets) via le port COM
     */
    bool sendCRSF(const std::array<uint8_t, CRSF::FRAME_LEN>& frame) {
        DWORD written = 0;
        BOOL ok = WriteFile(
            handle_,
            frame.data(),
            static_cast<DWORD>(frame.size()),
            &written,
            nullptr
        );
        return ok && (written == static_cast<DWORD>(frame.size()));
    }

private:
    HANDLE handle_ = INVALID_HANDLE_VALUE;
};

// ─────────────────────────────────────────────────────────────────────────────
//  IA — Détection d'obstacles et décision de vol
// ─────────────────────────────────────────────────────────────────────────────
class ObstacleDetector {
public:
    explicit ObstacleDetector(const std::string& modelPath = "") {
        if (!modelPath.empty()) {
            net_     = cv::dnn::readNetFromONNX(modelPath);
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            useYolo_ = true;
            std::cout << "[AI] YOLO ONNX charge : " << modelPath << "\n";
        } else {
            std::cout << "[AI] Mode detection de contours (Canny)\n";
        }
    }

    /**
     * detectObstacle() — analyse une frame, retourne une commande de vol
     *
     * Stratégie (mode Canny) :
     *   1. BGR → gris → flou gaussien → Canny
     *   2. Divise en 3 zones horizontales : gauche / centre / droite
     *   3. Densité de bords élevée au centre → obstacle
     *   4. Roll + Yaw vers la zone la moins chargée
     */
    FlightCommand detectObstacle(const cv::Mat& frame) {
        if (useYolo_) return detectYolo(frame);

        FlightCommand cmd;
        cv::Mat gray, edges;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, gray, {5, 5}, 0);
        cv::Canny(gray, edges, 50, 150);

        const int w  = edges.cols, h = edges.rows;
        const int zw = w / 3;

        // Densité de pixels actifs (0..1) dans une bande horizontale centrale
        auto density = [&](int x0, int x1) -> float {
            cv::Rect roi(x0, h / 4, x1 - x0, h / 2);
            return (float)cv::countNonZero(edges(roi)) /
                   (float)(roi.width * roi.height);
        };

        float dL = density(0,       zw);
        float dC = density(zw,  2 * zw);
        float dR = density(2 * zw,  w);

        cmd.obstacle = (dC > Config::OBSTACLE_THRESH);

        if (cmd.obstacle) {
            if (dL < dR) { cmd.roll = -0.6f; cmd.yaw = -0.3f; } // virer gauche
            else         { cmd.roll = +0.6f; cmd.yaw = +0.3f; } // virer droite
            cmd.pitch    = -0.2f;
            cmd.throttle =  0.55f;
        } else {
            cmd.pitch    =  0.2f;   // avancer
            cmd.throttle =  0.55f;
        }
        return cmd;
    }

private:
    cv::dnn::Net net_;
    bool         useYolo_ = false;

    FlightCommand detectYolo(const cv::Mat& frame) {
        FlightCommand cmd;
        cv::Mat blob = cv::dnn::blobFromImage(
            frame, 1.0 / 255.0, {640, 640}, {}, true, false);
        net_.setInput(blob);

        std::vector<cv::Mat> outs;
        net_.forward(outs, net_.getUnconnectedOutLayersNames());
        if (outs.empty()) return cmd;

        cv::Mat&  out     = outs[0];
        float     maxConf = 0.f;
        float     bestCx  = 0.5f;
        const int num     = out.size[2];

        for (int i = 0; i < num; i++) {
            float conf = out.at<float>(0, 4, i);
            if (conf > maxConf && conf > Config::YOLO_CONF) {
                maxConf = conf;
                bestCx  = out.at<float>(0, 0, i);
            }
        }

        cmd.obstacle = (maxConf > Config::YOLO_CONF);
        if (cmd.obstacle) {
            float offset  = bestCx - 0.5f;
            cmd.roll      = (offset > 0) ? -0.7f : +0.7f;
            cmd.yaw       = cmd.roll * 0.4f;
            cmd.pitch     = -0.15f;
            cmd.throttle  =  0.55f;
        } else {
            cmd.pitch    = 0.2f;
            cmd.throttle = 0.55f;
        }
        return cmd;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  CONVERSION FlightCommand → RCChannels
//  CH1=Roll  CH2=Pitch  CH3=Throttle  CH4=Yaw  CH5=Arming  CH6-16=neutres
// ─────────────────────────────────────────────────────────────────────────────
RCChannels commandToChannels(const FlightCommand& cmd, bool armed = false) {
    auto toUs = [](float v) -> uint16_t {
        if (v < -1.f) v = -1.f;
        if (v >  1.f) v =  1.f;
        return (uint16_t)(1500.f + v * 500.f);
    };
    auto thrUs = [](float v) -> uint16_t {
        if (v < 0.f) v = 0.f;
        if (v > 1.f) v = 1.f;
        return (uint16_t)(1000.f + v * 1000.f);
    };

    RCChannels ch;
    ch.fill(Config::RC_MID);
    ch[0] = toUs(cmd.roll);
    ch[1] = toUs(cmd.pitch);
    ch[2] = thrUs(cmd.throttle);
    ch[3] = toUs(cmd.yaw);
    ch[4] = armed ? (uint16_t)2000u : Config::RC_DISARM; // CH5 arming
    return ch;
}

// ─────────────────────────────────────────────────────────────────────────────
//  PRIORITÉ THREAD — Win32 (remplace SCHED_FIFO Linux)
// ─────────────────────────────────────────────────────────────────────────────
static void setWinThreadPriority(std::thread& t, int priority) {
    HANDLE h = reinterpret_cast<HANDLE>(t.native_handle());
    if (!SetThreadPriority(h, priority))
        std::cerr << "[RT] SetThreadPriority echoue (code " << GetLastError() << ")\n";
}

// ─────────────────────────────────────────────────────────────────────────────
//  CONTRÔLEUR PRINCIPAL — pipeline multi-thread
// ─────────────────────────────────────────────────────────────────────────────
class DroneController {
public:
    DroneController() : detector_(Config::YOLO_MODEL) {}

    void mainLoop() {
        running_ = true;

        // ── Ouverture UART ────────────────────────────────────────────────────
        try {
            uart_ = std::make_unique<UARTPort>(Config::UART_DEV, Config::UART_BAUD);
        } catch (const std::exception& e) {
            std::cerr << "[UART] " << e.what() << "\n";
            std::cerr << "[UART] Mode simulation — aucun paquet envoye\n";
        }

        // ── Ouverture caméra (DirectShow = backend natif Windows) ─────────────
        cap_.open(Config::CAM_INDEX); // sans forcer DirectShow
        if (!cap_.isOpened())
            cap_.open(Config::CAM_INDEX); // fallback sans backend forcé
        if (!cap_.isOpened()) {
            std::cerr << "[CAM] Camera introuvable (index "
                      << Config::CAM_INDEX << ")\n";
            running_ = false;
            return;
        }
        cap_.set(cv::CAP_PROP_FRAME_WIDTH,  Config::CAM_WIDTH);
        cap_.set(cv::CAP_PROP_FRAME_HEIGHT, Config::CAM_HEIGHT);
        cap_.set(cv::CAP_PROP_FPS,          Config::CAM_FPS);
        cap_.set(cv::CAP_PROP_BUFFERSIZE,   1); // réduit la latence

        std::cout << "[CTRL] Pipeline demarre — appuyer 'q' pour quitter\n";
        std::cout << "[CTRL] !! TESTER SANS HELICES !!\n";

        // ── Lancement des 3 threads ───────────────────────────────────────────
        auto t1 = std::thread([this] { captureThread(); });
        auto t2 = std::thread([this] { aiThread(); });
        auto t3 = std::thread([this] { crsfThread(); });

        // Priorités Win32 (pas besoin d'être admin contrairement à SCHED_FIFO)
        setWinThreadPriority(t3, THREAD_PRIORITY_TIME_CRITICAL); // CRSF max
        setWinThreadPriority(t2, THREAD_PRIORITY_HIGHEST);       // IA haute
        setWinThreadPriority(t1, THREAD_PRIORITY_ABOVE_NORMAL);  // Capture

        // ── Boucle UI principale (thread principal) ───────────────────────────
        while (running_) {
            VideoFrame vf;
            if (frameBuf_.popLatest(vf)) {
                FlightCommand lastCmd;
                if (cmdBuf_.popLatest(lastCmd)) {
                    std::string txt = lastCmd.obstacle
                        ? "OBSTACLE | Roll:" +
                          std::to_string((int)(lastCmd.roll * 100)) + "%"
                        : "VOIE LIBRE";
                    cv::putText(vf.img, txt, {10, 30},
                        cv::FONT_HERSHEY_SIMPLEX, 0.8,
                        lastCmd.obstacle ? cv::Scalar(0, 0, 255)
                                         : cv::Scalar(0, 255, 0), 2);
                }
                cv::imshow("FPV AI Controller", vf.img);
            }
            if (cv::waitKey(1) == 'q') running_ = false;
        }

        t1.join(); t2.join(); t3.join();
        cap_.release();
        cv::destroyAllWindows();
        std::cout << "[CTRL] Pipeline arrete proprement\n";
    }

private:
    // Buffers inter-threads
    CircularBuffer<VideoFrame,    Config::FRAME_BUF_SIZE> frameBuf_; // Capture → UI
    CircularBuffer<VideoFrame,    Config::FRAME_BUF_SIZE> aiIn_;     // Capture → IA
    CircularBuffer<FlightCommand, Config::CMD_BUF_SIZE>   cmdBuf_;   // IA → UI
    CircularBuffer<FlightCommand, Config::CMD_BUF_SIZE>   crsfIn_;   // IA → CRSF

    std::atomic<bool>         running_{false};
    cv::VideoCapture          cap_;
    std::unique_ptr<UARTPort> uart_;
    ObstacleDetector          detector_;

    // ── Thread 1 : Capture ───────────────────────────────────────────────────
    void captureThread() {
        std::cout << "[CAP] Thread demarre\n";
        while (running_) {
            VideoFrame vf;
            if (!cap_.read(vf.img) || vf.img.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }
            vf.ts = std::chrono::steady_clock::now();
            frameBuf_.push(VideoFrame{vf.img.clone(), vf.ts}); // vers UI
            aiIn_.push(std::move(vf));                          // vers IA
        }
        std::cout << "[CAP] Thread termine\n";
    }

    // ── Thread 2 : IA / Décision ─────────────────────────────────────────────
    void aiThread() {
        std::cout << "[AI] Thread demarre\n";
        while (running_) {
            VideoFrame vf;
            if (!aiIn_.popLatest(vf)) {
                std::this_thread::sleep_for(std::chrono::microseconds(500));
                continue;
            }
            auto t0  = std::chrono::steady_clock::now();
            auto cmd = detector_.detectObstacle(vf.img);
            auto dt  = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::steady_clock::now() - t0).count();
            if (dt > Config::AI_MAX_MS)
                std::cerr << "[AI] Latence " << dt << " ms (budget "
                          << Config::AI_MAX_MS << " ms)\n";

            cmdBuf_.push(FlightCommand{cmd}); // vers UI
            crsfIn_.push(std::move(cmd));     // vers CRSF
        }
        std::cout << "[AI] Thread termine\n";
    }

    // ── Thread 3 : Envoi CRSF @ 250 Hz ──────────────────────────────────────
    void crsfThread() {
        std::cout << "[CRSF] Thread demarre @ "
                  << Config::CRSF_FREQ_HZ << " Hz\n";

        // Améliore la résolution du timer Windows (défaut : ~15ms → 1ms)
        timeBeginPeriod(1);

        constexpr auto period = std::chrono::microseconds(
            1'000'000 / Config::CRSF_FREQ_HZ);
        auto next = std::chrono::steady_clock::now();

        FlightCommand lastCmd;
        auto          lastCmdTs = std::chrono::steady_clock::now();

        while (running_) {
            FlightCommand cmd;
            if (crsfIn_.popLatest(cmd)) {
                lastCmd   = cmd;
                lastCmdTs = std::chrono::steady_clock::now();
            }

            // FAILSAFE : > 500 ms sans commande → throttle 0
            auto ageMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - lastCmdTs).count();
            if (ageMs > 500) {
                lastCmd          = {};
                lastCmd.throttle = 0.f;
            }

            auto frame = CRSF::packCRSF(
                commandToChannels(lastCmd, /*armed=*/false));
            if (uart_) uart_->sendCRSF(frame);

            next += period;
            std::this_thread::sleep_until(next);
        }

        // Frame failsafe finale avant arrêt
        RCChannels safe;
        safe.fill(Config::RC_MID);
        safe[2] = Config::RC_MIN;
        safe[4] = Config::RC_DISARM;
        if (uart_) uart_->sendCRSF(CRSF::packCRSF(safe));

        timeEndPeriod(1);
        std::cout << "[CRSF] Thread termine\n";
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  MAIN
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    std::cout << "=== DEMARRAGE ===" << std::endl; 
    std::flush(std::cout);
    std::cout << "=== FPV DRONE AI CONTROLLER (Windows) ===\n";
    std::cout << "!! RETIRER LES HELICES AVANT TOUT TEST !!\n\n";

    std::cout << "Camera  : index " << Config::CAM_INDEX
              << "  " << Config::CAM_WIDTH << "x" << Config::CAM_HEIGHT
              << " @" << Config::CAM_FPS << " fps\n";
    std::cout << "UART    : " << Config::UART_DEV
              << "  @" << Config::UART_BAUD << " baud\n";
    std::cout << "CRSF    : " << Config::CRSF_FREQ_HZ << " Hz\n";
    std::cout << "IA      : "
              << (strlen(Config::YOLO_MODEL) > 0
                      ? Config::YOLO_MODEL
                      : "Canny (detection de contours)")
              << "\n\n";

    // Test unitaire packCRSF au démarrage
    {
        RCChannels test;
        test.fill(1500);
        auto f = CRSF::packCRSF(test);
        std::cout << "[TEST] Frame CRSF neutre (hex) : ";
        for (auto b : f) printf("%02X ", b);
        std::cout << "\n\n";
    }

    DroneController ctrl;
    ctrl.mainLoop();
    return 0;
}

/*
 * ============================================================
 *  GUIDE DE COMPILATION WINDOWS
 * ============================================================
 *
 *  OPTION A — MSYS2 / MinGW-w64 (recommandé, gratuit)
 *  ──────────────────────────────────────────────────
 *  1. Installer MSYS2 : https://www.msys2.org/
 *  2. Dans le terminal "MSYS2 MinGW 64-bit" :
 *       pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-opencv
 *  3. Compiler :
 *       g++ -O2 -std=c++17 fpv_drone_ai_controller.cpp \
 *           $(pkg-config --cflags --libs opencv4) \
 *           -lwinmm \
 *           -o fpv_drone_ai_controller.exe
 *
 *  OPTION B — OpenCV précompilé officiel + MinGW
 *  ──────────────────────────────────────────────
 *  (Télécharger depuis https://opencv.org/releases/ → Windows)
 *
 *       g++ -O2 -std=c++17 fpv_drone_ai_controller.cpp \
 *           -IC:\opencv\build\include \
 *           -LC:\opencv\build\x64\mingw\lib \
 *           -lopencv_world4100 \
 *           -lwinmm \
 *           -o fpv_drone_ai_controller.exe
 *
 *  NOTE : adapter "4100" à la version installée (ex. 4100 = 4.10.0)
 *
 *  OPTION C — Visual Studio (MSVC)
 *  ─────────────────────────────────
 *  1. Créer un projet "Application console C++"
 *  2. Propriétés → C/C++ → Répertoires include → ajouter C:\opencv\build\include
 *  3. Propriétés → Éditeur de liens → Entrée → Dépendances supplémentaires :
 *       opencv_world4100.lib ; winmm.lib
 *  4. Propriétés → Éditeur de liens → Général → Répertoires bibliothèques :
 *       C:\opencv\build\x64\vc16\lib
 *  5. Copier opencv_world4100.dll dans le dossier du .exe
 *
 * ============================================================
 *  TROUVER LE BON PORT COM
 * ============================================================
 *  Gestionnaire de périphériques → Ports (COM et LPT)
 *  Le module ELRS USB apparaît souvent comme :
 *    - "Silicon Labs CP210x" (UART) → COMx
 *    - "CH340" (clones) → COMx
 *  Modifier Config::UART_DEV, exemples :
 *    "\\\\.\\COM3"    "\\\\.\\COM10"    "\\\\.\\COM21"
 *
 * ============================================================
 *  TROUVER L'INDEX DE LA CAMÉRA DE CAPTURE FPV
 * ============================================================
 *  Tester les index 0, 1, 2... dans Config::CAM_INDEX.
 *  Ou lancer ce mini-programme de test séparé :
 *
 *    for (int i = 0; i < 5; i++) {
 *        cv::VideoCapture c(i, cv::CAP_DSHOW);
 *        if (c.isOpened())
 *            std::cout << "Camera trouvee : index " << i << "\n";
 *    }
 *
 * ============================================================
 *  TEST DE SÉCURITÉ ÉTAPE PAR ÉTAPE
 * ============================================================
 *  1. Lancer sans module ELRS → doit afficher "Mode simulation"
 *  2. Brancher le module ELRS → doit afficher "[UART] COM3 ouvert"
 *  3. Vérifier les octets dans PuTTY (Serial, 420000 baud, 8N1)
 *     → doit voir "C8 18 16 ..." en continu
 *  4. Ouvrir Betaflight Configurator → Receiver
 *     → CH1-4 bougent selon l'image caméra
 *     → CH5 = 1000 µs → drone désarmé (SÉCURITÉ)
 *  5. Seulement si tout valide : passer armed=true dans
 *     commandToChannels() pour autoriser l'armement
 * ============================================================
 */
