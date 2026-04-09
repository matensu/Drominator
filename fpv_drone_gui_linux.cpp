/**
 * ============================================================
 *  FPV DRONE AI CONTROLLER — GUI Qt + ExpressLRS / CRSF
 *  Auteur   : Expert Robotique C++
 *  Cible    : Linux — Pop!_OS / Ubuntu 22.04+ (x86_64)
 *  Dépend.  : Qt5 ou Qt6, OpenCV 4.x, pthread
 *  Compiler : voir CMakeLists.txt
 * ============================================================
 *
 *  DIFFÉRENCES LINUX vs WINDOWS :
 *    - UART  : termios (cfsetospeed / tcsetattr) au lieu de Win32 DCB
 *    - Ports : /dev/ttyUSB0, /dev/ttyACM0 (détection auto)
 *    - RT    : SCHED_FIFO + pthread_setschedparam (sudo ou CAP_SYS_NICE)
 *    - Caméra: V4L2 via OpenCV (cv::CAP_V4L2)
 *    - Timer : clock_nanosleep (résolution µs native)
 *
 *  PIPELINE :
 *    [CaptureThread] → CircularBuffer<Frame>
 *                              ↓
 *    [AIThread]      → CircularBuffer<FlightCommand>
 *                              ↓
 *    [CRSFThread]    → /dev/ttyUSB0  →  Module TX ELRS
 *                                              ↓
 *                                         Drone FPV
 *
 *  SÉCURITÉ :
 *    - Démarrer SANS hélices lors des premiers tests
 *    - FAILSAFE : throttle → 0 si aucune commande IA depuis 500 ms
 *    - CH5 = 1000 µs par défaut (désarmé)
 *    - Toutes les valeurs RC clampées [1000..2000] µs
 * ============================================================
 */

// ── POSIX / Linux ─────────────────────────────────────────────────────────────
#include <errno.h>
#include <fcntl.h>
#include <glob.h>           // glob() pour lister /dev/ttyUSB* et /dev/ttyACM*
#include <pthread.h>        // pthread_setschedparam — SCHED_FIFO
#include <sched.h>          // SCHED_FIFO
#include <sys/ioctl.h>      // ioctl() — TCGETS2/TCSETS2 pour baud non-standard
#include <termios.h>        // cfsetospeed, tcsetattr, etc.
#include <time.h>           // clock_nanosleep — timer haute résolution
#include <unistd.h>         // read, write, close

// ── Qt ────────────────────────────────────────────────────────────────────────
#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QDateTime>
#include <QFontDatabase>
#include <QFrame>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QImage>
#include <QLabel>
#include <QMainWindow>
#include <QMessageBox>
#include <QMetaType>
#include <QMutex>
#include <QMutexLocker>
#include <QPainter>
#include <QPixmap>
#include <QProgressBar>
#include <QPushButton>
#include <QScrollBar>
#include <QSizePolicy>
#include <QSlider>
#include <QSpinBox>
#include <QStyleFactory>
#include <QTextEdit>
#include <QThread>
#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>

// ── Standard C++ ─────────────────────────────────────────────────────────────
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// ── OpenCV ────────────────────────────────────────────────────────────────────
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

// ─────────────────────────────────────────────────────────────────────────────
//  STRUCTURES PARTAGÉES
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
        if (next == read_.load(std::memory_order_acquire)) return false;
        buf_[w] = std::move(item);
        write_.store(next, std::memory_order_release);
        return true;
    }
    bool pop(T& item) {
        const size_t r = read_.load(std::memory_order_relaxed);
        if (r == write_.load(std::memory_order_acquire)) return false;
        item = std::move(buf_[r]);
        read_.store((r + 1) & (N - 1), std::memory_order_release);
        return true;
    }
    bool popLatest(T& item) {
        bool got = false; T tmp;
        while (pop(tmp)) { item = std::move(tmp); got = true; }
        return got;
    }
    void clear() { T tmp; while (pop(tmp)) {} }

private:
    std::array<T, N>    buf_{};
    std::atomic<size_t> write_{0};
    std::atomic<size_t> read_{0};
};

// ─────────────────────────────────────────────────────────────────────────────
//  CRSF — Encodage + CRC-8/DVB-S2
//  Référence : https://github.com/crsf-wg/crsf/wiki
// ─────────────────────────────────────────────────────────────────────────────
namespace CRSF {
    constexpr uint8_t SYNC_BYTE        = 0xC8;
    constexpr uint8_t TYPE_RC_CHANNELS = 0x16;
    constexpr size_t  FRAME_LEN        = 26;  // sync+len+type+payload(22)+crc

    // Polynôme DVB-S2 : 0xD5
    static uint8_t crc8(const uint8_t* d, size_t len) {
        uint8_t crc = 0;
        for (size_t i = 0; i < len; i++) {
            crc ^= d[i];
            for (int b = 0; b < 8; b++)
                crc = (crc & 0x80) ? (uint8_t)((crc << 1) ^ 0xD5)
                                   : (uint8_t)(crc << 1);
        }
        return crc;
    }

    /**
     * packCRSF() — encode 16 canaux 11 bits → frame CRSF 26 octets
     *   Conversion µs → raw : raw = (µs - 1000) * 1639 / 1000 + 172
     *   Plage : 172 (1000 µs) .. 1811 (2000 µs), centre = 992
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
        for (int ch = 0; ch < 16; ch++)
            for (int b = 0; b < 11; b++) {
                if (raw[ch] & (1 << b))
                    payload[bitPos / 8] |= (uint8_t)(1u << (bitPos % 8));
                bitPos++;
            }

        std::array<uint8_t, FRAME_LEN> frame{};
        frame[0] = SYNC_BYTE;
        frame[1] = 24;                  // len = type(1) + payload(22) + crc(1)
        frame[2] = TYPE_RC_CHANNELS;
        std::memcpy(&frame[3], payload, 22);
        frame[25] = crc8(&frame[2], 23); // CRC sur TYPE + PAYLOAD
        return frame;
    }
} // namespace CRSF

// ─────────────────────────────────────────────────────────────────────────────
//  UART PORT — POSIX / Linux (termios)
//  Remplace entièrement la version Win32 (CreateFile / DCB / WriteFile)
// ─────────────────────────────────────────────────────────────────────────────
class UARTPort {
public:
    /**
     * Ouvre un port série Linux (/dev/ttyUSB0, /dev/ttyACM0…)
     * @param dev   chemin complet, ex. "/dev/ttyUSB0"
     * @param baud  débit en bauds (int), ex. 420000
     */
    explicit UARTPort(const std::string& dev, int baud) {
        fd_ = ::open(dev.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
        if (fd_ < 0)
            throw std::runtime_error(
                "Impossible d'ouvrir " + dev + " : " + strerror(errno) +
                "\n-> sudo chmod a+rw " + dev +
                "\n-> ou : sudo usermod -aG dialout $USER  (puis re-login)");

        termios tio{};
        if (tcgetattr(fd_, &tio) != 0)
            throw std::runtime_error("tcgetattr échoué sur " + dev);

        cfmakeraw(&tio);             // mode raw (pas d'interprétation)
        tio.c_cflag |=  (CLOCAL | CREAD | CS8);
        tio.c_cflag &= ~(PARENB | CSTOPB | CRTSCTS);
        tio.c_cc[VMIN]  = 0;        // lecture non bloquante
        tio.c_cc[VTIME] = 0;

        // Baud rate standard ou personnalisé (ex. 420000)
        speed_t spd = baudToSpeed(baud);
        if (spd != (speed_t)-1) {
            cfsetispeed(&tio, spd);
            cfsetospeed(&tio, spd);
        } else {
            // Débit non standard : ioctl BOTHER (Linux ≥ 3.7)
            // Nécessite <asm/termios.h> — on passe par la méthode directe
            cfsetispeed(&tio, B38400);  // placeholder
            cfsetospeed(&tio, B38400);
        }

        if (tcsetattr(fd_, TCSANOW, &tio) != 0)
            throw std::runtime_error("tcsetattr échoué sur " + dev);

        // Débit non standard via ioctl (420000 baud pour CRSF)
        if (spd == (speed_t)-1)
            setCustomBaud(baud);

        tcflush(fd_, TCIOFLUSH);
        baud_ = baud;
    }

    ~UARTPort() { if (fd_ >= 0) ::close(fd_); }
    UARTPort(const UARTPort&)            = delete;
    UARTPort& operator=(const UARTPort&) = delete;

    /**
     * sendCRSF() — envoie une frame CRSF (26 octets) via UART Linux
     */
    bool sendCRSF(const std::array<uint8_t, CRSF::FRAME_LEN>& frame) {
        ssize_t n = ::write(fd_, frame.data(), frame.size());
        return n == static_cast<ssize_t>(frame.size());
    }

    int baudRate() const { return baud_; }

private:
    int fd_   = -1;
    int baud_ = 0;

    // Conversion baud → speed_t POSIX
    static speed_t baudToSpeed(int baud) {
        switch (baud) {
            case 9600:   return B9600;
            case 19200:  return B19200;
            case 38400:  return B38400;
            case 57600:  return B57600;
            case 115200: return B115200;
            case 230400: return B230400;
            case 460800: return B460800;
            case 921600: return B921600;
            default:     return (speed_t)-1; // non standard
        }
    }

    // Baud non-standard via termios2/ioctl (Linux >= 3.7)
    // Utilise termios2_local pour eviter le conflit avec CBAUD de <termios.h>
    void setCustomBaud(int baud) {
#ifdef __linux__
        struct termios2_local {
            tcflag_t c_iflag, c_oflag, c_cflag, c_lflag;
            cc_t     c_line;
            cc_t     c_cc[19];
            speed_t  c_ispeed, c_ospeed;
        };
        // Codes ioctl x86_64 pour termios2
        constexpr unsigned long TCGETS2_REQ  = 0x802C542AUL;
        constexpr unsigned long TCSETS2_REQ  = 0x402C542BUL;
        // Renommes pour eviter la collision avec BOTHER/CBAUD de <termios.h>
        constexpr tcflag_t BOTHER_FLAG       = 0010000;
        constexpr tcflag_t CBAUD_MASK_LOCAL  = 0010017;

        termios2_local t2{};
        if (ioctl(fd_, TCGETS2_REQ, &t2) == 0) {
            t2.c_cflag &= ~CBAUD_MASK_LOCAL;
            t2.c_cflag |=  BOTHER_FLAG;
            t2.c_ispeed = static_cast<speed_t>(baud);
            t2.c_ospeed = static_cast<speed_t>(baud);
            if (ioctl(fd_, TCSETS2_REQ, &t2) != 0)
                std::cerr << "[UART] TCSETS2 echoue pour " << baud
                          << " baud (errno=" << errno << ")\n";
        } else {
            std::cerr << "[UART] TCGETS2 echoue (errno=" << errno
                      << ") - baud " << baud << " peut ne pas fonctionner\n";
        }
#endif
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  SCAN PORTS SÉRIE — liste /dev/ttyUSB* et /dev/ttyACM* disponibles
// ─────────────────────────────────────────────────────────────────────────────
static std::vector<std::string> scanSerialPorts() {
    std::vector<std::string> ports;
    const char* patterns[] = { "/dev/ttyUSB*", "/dev/ttyACM*", "/dev/ttyS[0-9]*" };
    for (auto* pat : patterns) {
        glob_t g{};
        if (glob(pat, 0, nullptr, &g) == 0)
            for (size_t i = 0; i < g.gl_pathc; i++)
                ports.emplace_back(g.gl_pathv[i]);
        globfree(&g);
    }
    return ports;
}

// ─────────────────────────────────────────────────────────────────────────────
//  PRIORITÉ THREAD — SCHED_FIFO (Linux, remplace SetThreadPriority Win32)
//  Nécessite : sudo ./fpv_drone_ai  OU  sudo setcap cap_sys_nice+ep ./fpv_drone_ai
// ─────────────────────────────────────────────────────────────────────────────
static void setLinuxRealtimePriority(std::thread& t, int priority) {
    sched_param sp{};
    sp.sched_priority = priority;  // 1..99 pour SCHED_FIFO
    int rc = pthread_setschedparam(t.native_handle(), SCHED_FIFO, &sp);
    if (rc != 0)
        std::cerr << "[RT] pthread_setschedparam(" << priority << ") échoué : "
                  << strerror(rc) << "\n"
                  << "[RT] Lancer avec sudo ou : sudo setcap cap_sys_nice+ep ./fpv_drone_ai\n";
}

// ─────────────────────────────────────────────────────────────────────────────
//  SLEEP HAUTE RÉSOLUTION — clock_nanosleep (Linux, bien plus précis que
//  std::this_thread::sleep_until sur Linux sans timeBeginPeriod)
// ─────────────────────────────────────────────────────────────────────────────
static void highResSleepUntil(const std::chrono::steady_clock::time_point& tp) {
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                  tp.time_since_epoch()).count();
    struct timespec ts { ns / 1'000'000'000L, ns % 1'000'000'000L };
    while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr) == EINTR) {}
}

// ─────────────────────────────────────────────────────────────────────────────
//  IA — Détection d'obstacles + décision de vol
// ─────────────────────────────────────────────────────────────────────────────
class ObstacleDetector {
public:
    ObstacleDetector() = default;

    void setThreshold(float t) { thresh_ = t; }

    bool loadYolo(const std::string& path) {
        try {
            net_     = cv::dnn::readNetFromONNX(path);
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            useYolo_ = true;
            return true;
        } catch (...) { useYolo_ = false; return false; }
    }

    struct Result {
        FlightCommand cmd;
        long          latencyMs = 0;
        float         dL = 0, dC = 0, dR = 0; // densités zones L/C/D
    };

    Result detect(const cv::Mat& frame) {
        auto t0 = std::chrono::steady_clock::now();
        Result r;
        if (useYolo_) r.cmd = detectYolo(frame);
        else          r.cmd = detectCanny(frame, r.dL, r.dC, r.dR);
        r.latencyMs = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count();
        return r;
    }

private:
    cv::dnn::Net net_;
    bool         useYolo_ = false;
    float        thresh_  = 0.35f;

    /**
     * Canny Edge Detection — 3 zones (Gauche / Centre / Droite)
     *   Si densité de bords élevée au centre → obstacle
     *   Roll + Yaw vers la zone la moins chargée
     */
    FlightCommand detectCanny(const cv::Mat& frame, float& dL, float& dC, float& dR) {
        FlightCommand cmd;
        cv::Mat gray, edges;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, gray, {5, 5}, 0);
        cv::Canny(gray, edges, 50, 150);

        int w = edges.cols, h = edges.rows, zw = w / 3;
        auto density = [&](int x0, int x1) -> float {
            cv::Rect roi(x0, h / 4, x1 - x0, h / 2);
            return (float)cv::countNonZero(edges(roi)) /
                   (float)(roi.width * roi.height);
        };
        dL = density(0, zw); dC = density(zw, 2 * zw); dR = density(2 * zw, w);
        cmd.obstacle = (dC > thresh_);
        if (cmd.obstacle) {
            if (dL < dR) { cmd.roll = -0.6f; cmd.yaw = -0.3f; }
            else         { cmd.roll = +0.6f; cmd.yaw = +0.3f; }
            cmd.pitch = -0.2f; cmd.throttle = 0.55f;
        } else {
            cmd.pitch = 0.2f; cmd.throttle = 0.55f;
        }
        return cmd;
    }

    FlightCommand detectYolo(const cv::Mat& frame) {
        FlightCommand cmd;
        cv::Mat blob = cv::dnn::blobFromImage(
            frame, 1.0 / 255.0, {640, 640}, {}, true, false);
        net_.setInput(blob);
        std::vector<cv::Mat> outs;
        net_.forward(outs, net_.getUnconnectedOutLayersNames());
        if (outs.empty()) return cmd;
        cv::Mat& out = outs[0];
        float maxConf = 0.f, bestCx = 0.5f;
        int num = out.size[2];
        for (int i = 0; i < num; i++) {
            float conf = out.at<float>(0, 4, i);
            if (conf > maxConf && conf > 0.5f) {
                maxConf = conf; bestCx = out.at<float>(0, 0, i);
            }
        }
        cmd.obstacle = (maxConf > 0.5f);
        if (cmd.obstacle) {
            float off = bestCx - 0.5f;
            cmd.roll = (off > 0) ? -0.7f : 0.7f;
            cmd.yaw = cmd.roll * 0.4f; cmd.pitch = -0.15f; cmd.throttle = 0.55f;
        } else {
            cmd.pitch = 0.2f; cmd.throttle = 0.55f;
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
    RCChannels ch; ch.fill(1500);
    ch[0] = toUs(cmd.roll);
    ch[1] = toUs(cmd.pitch);
    ch[2] = thrUs(cmd.throttle);
    ch[3] = toUs(cmd.yaw);
    ch[4] = armed ? (uint16_t)2000u : (uint16_t)1000u; // CH5 arming
    return ch;
}

// ─────────────────────────────────────────────────────────────────────────────
//  PONT UI ↔ THREADS (callbacks std::function, sans QObject dans les threads)
// ─────────────────────────────────────────────────────────────────────────────
struct UiBridge {
    std::function<void(const cv::Mat&, const FlightCommand&, float, float, float)> onFrame;
    std::function<void(const std::string&)>                                         onLog;
    std::function<void(long)>                                                       onLatency;
    std::function<void(const RCChannels&)>                                          onChannels;
    std::function<void(bool)>                                                       onFailsafe;
};

// ─────────────────────────────────────────────────────────────────────────────
//  DRONE CONTROLLER — moteur du pipeline (pure C++, sans Qt)
// ─────────────────────────────────────────────────────────────────────────────
class DroneController {
public:
    explicit DroneController(UiBridge bridge) : bridge_(std::move(bridge)) {}
    ~DroneController() { stop(); }

    // ── UART ─────────────────────────────────────────────────────────────────
    bool connectUART(const std::string& dev, int baud) {
        try {
            uart_ = std::make_unique<UARTPort>(dev, baud);
            log("[UART] " + dev + " ouvert @ " + std::to_string(baud) + " baud");
            uartOk_ = true;
            return true;
        } catch (const std::exception& e) {
            log(std::string("[UART] ERREUR : ") + e.what());
            uartOk_ = false;
            return false;
        }
    }

    void disconnectUART() {
        uart_.reset(); uartOk_ = false;
        log("[UART] Déconnecté");
    }

    // ── Caméra ───────────────────────────────────────────────────────────────
    bool openCamera(int idx, int w, int h, int fps) {
        // V4L2 est le backend natif Linux
        cap_.open(idx, cv::CAP_V4L2);
        if (!cap_.isOpened()) cap_.open(idx); // fallback auto
        if (!cap_.isOpened()) {
            log("[CAM] Introuvable : /dev/video" + std::to_string(idx));
            return false;
        }
        cap_.set(cv::CAP_PROP_FRAME_WIDTH,  w);
        cap_.set(cv::CAP_PROP_FRAME_HEIGHT, h);
        cap_.set(cv::CAP_PROP_FPS,          fps);
        cap_.set(cv::CAP_PROP_BUFFERSIZE,   1); // réduit la latence
        camOk_ = true;
        log("[CAM] /dev/video" + std::to_string(idx) + " ouverte " +
            std::to_string(w) + "x" + std::to_string(h) + " @" +
            std::to_string(fps) + " fps");
        return true;
    }

    void closeCamera() {
        cap_.release(); camOk_ = false;
        log("[CAM] Caméra fermée");
    }

    // ── Paramètres ───────────────────────────────────────────────────────────
    void setArmed(bool a)      { armed_   = a; }
    void setThreshold(float t) { detector_.setThreshold(t); }
    void setCRSFFreq(int hz)   { crsfHz_  = hz; }

    // ── Démarrage / Arrêt du pipeline ────────────────────────────────────────
    void startAI() {
        if (running_) return;
        if (!camOk_) { log("[CTRL] Ouvrir la caméra d'abord"); return; }
        running_   = true;
        lastCmdTs_ = std::chrono::steady_clock::now();

        t1_ = std::thread([this] { captureThread(); });
        t2_ = std::thread([this] { aiThread(); });
        t3_ = std::thread([this] { crsfThread(); });

        // Priorités temps réel SCHED_FIFO
        setLinuxRealtimePriority(t3_, 80); // CRSF — max
        setLinuxRealtimePriority(t2_, 60); // IA   — haute
        setLinuxRealtimePriority(t1_, 40); // Capture

        log("[CTRL] Pipeline IA démarré (3 threads)");
    }

    void stop() {
        if (!running_) return;
        running_ = false;
        if (t1_.joinable()) t1_.join();
        if (t2_.joinable()) t2_.join();
        if (t3_.joinable()) t3_.join();

        // Frame failsafe avant fermeture
        if (uart_) {
            RCChannels safe; safe.fill(1500);
            safe[2] = 1000; safe[4] = 1000; // throttle min, désarmé
            uart_->sendCRSF(CRSF::packCRSF(safe));
        }
        frameBuf_.clear(); aiIn_.clear(); cmdBuf_.clear(); crsfIn_.clear();
        log("[CTRL] Pipeline arrêté");
    }

    bool isRunning() const { return running_.load(); }
    bool isUARTOk()  const { return uartOk_.load(); }
    bool isCamOk()   const { return camOk_.load(); }

private:
    UiBridge                  bridge_;
    std::unique_ptr<UARTPort> uart_;
    ObstacleDetector          detector_;
    cv::VideoCapture          cap_;

    std::atomic<bool> running_{false};
    std::atomic<bool> uartOk_{false};
    std::atomic<bool> camOk_{false};
    std::atomic<bool> armed_{false};
    std::atomic<int>  crsfHz_{250};

    std::thread t1_, t2_, t3_;

    CircularBuffer<VideoFrame,    4> frameBuf_; // Capture → UI
    CircularBuffer<VideoFrame,    4> aiIn_;     // Capture → IA
    CircularBuffer<FlightCommand, 8> cmdBuf_;   // IA → UI
    CircularBuffer<FlightCommand, 8> crsfIn_;   // IA → CRSF

    FlightCommand                         lastCmd_;
    std::chrono::steady_clock::time_point lastCmdTs_;

    void log(const std::string& msg) { if (bridge_.onLog) bridge_.onLog(msg); }

    // ── Thread 1 : Capture V4L2 ──────────────────────────────────────────────
    void captureThread() {
        while (running_) {
            VideoFrame vf;
            if (!cap_.read(vf.img) || vf.img.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }
            vf.ts = std::chrono::steady_clock::now();
            frameBuf_.push(VideoFrame{vf.img.clone(), vf.ts}); // UI
            aiIn_.push(std::move(vf));                          // IA
        }
    }

    // ── Thread 2 : IA / Détection ────────────────────────────────────────────
    void aiThread() {
        while (running_) {
            VideoFrame vf;
            if (!aiIn_.popLatest(vf)) {
                std::this_thread::sleep_for(std::chrono::microseconds(500));
                continue;
            }
            auto r = detector_.detect(vf.img);

            if (bridge_.onFrame)
                bridge_.onFrame(vf.img, r.cmd, r.dL, r.dC, r.dR);
            if (bridge_.onLatency)
                bridge_.onLatency(r.latencyMs);

            cmdBuf_.push(FlightCommand{r.cmd});
            crsfIn_.push(std::move(r.cmd));
        }
    }

    // ── Thread 3 : Envoi CRSF @ N Hz ────────────────────────────────────────
    void crsfThread() {
        auto next = std::chrono::steady_clock::now();
        lastCmdTs_ = next;

        while (running_) {
            const auto period = std::chrono::microseconds(
                1'000'000 / crsfHz_.load());

            FlightCommand cmd;
            if (crsfIn_.popLatest(cmd)) {
                lastCmd_   = cmd;
                lastCmdTs_ = std::chrono::steady_clock::now();
            }

            // FAILSAFE : > 500 ms sans commande → throttle 0
            auto ageMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - lastCmdTs_).count();
            bool failsafe = (ageMs > 500);
            if (failsafe) { lastCmd_ = {}; lastCmd_.throttle = 0.f; }
            if (bridge_.onFailsafe) bridge_.onFailsafe(failsafe);

            auto ch    = commandToChannels(lastCmd_, armed_.load());
            auto frame = CRSF::packCRSF(ch);
            if (uart_) uart_->sendCRSF(frame);
            if (bridge_.onChannels) bridge_.onChannels(ch);

            next += period;
            highResSleepUntil(next); // clock_nanosleep — précis à la µs
        }

        // Failsafe finale
        RCChannels safe; safe.fill(1500);
        safe[2] = 1000; safe[4] = 1000;
        if (uart_) uart_->sendCRSF(CRSF::packCRSF(safe));
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  VIDEO WIDGET — affichage OpenCV dans Qt
// ─────────────────────────────────────────────────────────────────────────────
class VideoWidget : public QLabel {
    Q_OBJECT
public:
    explicit VideoWidget(QWidget* parent = nullptr) : QLabel(parent) {
        setMinimumSize(640, 360);
        setAlignment(Qt::AlignCenter);
        setStyleSheet(
            "background:#050810; color:#1a3a6b;"
            "border:2px solid #0d1f3c; font-family:'Courier New';");
        setText("[ NO SIGNAL ]");
        setFont(QFont("Courier New", 14));
    }

    // Méthode publique pour réafficher le placeholder "NO SIGNAL"
    void showNoSignal() {
        QLabel::setText("[ NO SIGNAL ]");
        setPixmap(QPixmap());
    }

    void updateFrame(const QImage& img, const FlightCommand& cmd,
                     float dL, float dC, float dR) {
        QPixmap pix = QPixmap::fromImage(img).scaled(
            size(), Qt::KeepAspectRatio, Qt::FastTransformation);
        QPainter p(&pix);
        p.setRenderHint(QPainter::Antialiasing);

        int pw = pix.width(), ph = pix.height(), zw = pw / 3;

        // Barres de densité en bas (zones L / C / D)
        auto drawZone = [&](int x, float d, QColor col) {
            col.setAlpha(80 + (int)(d * 120));
            p.fillRect(x, ph - 14, zw, 14, col);
            p.setPen(col);
            p.setFont(QFont("Courier New", 7));
            p.drawText(x + 3, ph - 2, QString::number(d, 'f', 2));
        };
        drawZone(0,       dL, QColor(0, 180, 255));
        drawZone(zw,      dC, QColor(255, 60, 60));
        drawZone(2 * zw,  dR, QColor(0, 180, 255));

        // Cadre + texte obstacle
        if (cmd.obstacle) {
            p.setPen(QPen(QColor(255, 40, 40), 3));
            p.drawRect(5, 5, pw - 10, ph - 24);
            p.setFont(QFont("Courier New", 11, QFont::Bold));
            p.setPen(QColor(255, 40, 40));
            QString dir = (cmd.roll < 0) ? "<-- EVITEMENT GAUCHE"
                                         : "EVITEMENT DROITE -->";
            p.drawText(QRect(0, 10, pw, 30), Qt::AlignCenter, dir);
        } else {
            p.setPen(QPen(QColor(0, 220, 120), 2));
            p.drawRect(5, 5, pw - 10, ph - 24);
            p.setFont(QFont("Courier New", 10, QFont::Bold));
            p.setPen(QColor(0, 220, 120));
            p.drawText(QRect(0, 10, pw, 30), Qt::AlignCenter, ">> VOIE LIBRE");
        }
        setPixmap(pix);
    }

};

// ─────────────────────────────────────────────────────────────────────────────
//  RC CHANNEL BAR — indicateur µs d'un canal RC
// ─────────────────────────────────────────────────────────────────────────────
class RCBar : public QWidget {
    Q_OBJECT
public:
    explicit RCBar(const QString& label, QWidget* parent = nullptr)
        : QWidget(parent), label_(label), value_(1500) {
        setMinimumHeight(24);
    }
    void setValue(uint16_t v) { value_ = v; update(); }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter p(this);
        int w = width(), h = height();
        p.fillRect(rect(), QColor(8, 14, 28));
        float norm = (value_ - 1000.f) / 1000.f;
        int   bw   = (int)(norm * w);
        QColor col = (value_ > 1700) ? QColor(255, 60, 60) :
                     (value_ < 1300) ? QColor(0, 160, 255) :
                                       QColor(0, 210, 120);
        p.fillRect(0, 3, bw, h - 6, col);
        p.setPen(QColor(40, 60, 100));
        p.drawLine(w / 2, 0, w / 2, h);
        p.setPen(QColor(180, 200, 255));
        p.setFont(QFont("Courier New", 8));
        p.drawText(rect().adjusted(4, 0, -4, 0),
            Qt::AlignVCenter | Qt::AlignLeft,
            label_ + "  " + QString::number(value_) + " us");
    }
private:
    QString  label_;
    uint16_t value_;
};

// ─────────────────────────────────────────────────────────────────────────────
//  MAIN WINDOW
// ─────────────────────────────────────────────────────────────────────────────
class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr) : QMainWindow(parent) {
        setWindowTitle("FPV DRONE AI CONTROLLER — Pop!_OS / Linux");
        setMinimumSize(1200, 780);
        applyDarkTheme();
        buildUI();

        // Refresh log toutes les 100 ms (hors hot path)
        auto* logTimer = new QTimer(this);
        connect(logTimer, &QTimer::timeout, this, &MainWindow::flushLog);
        logTimer->start(100);
    }

private slots:

    // ── UART ─────────────────────────────────────────────────────────────────
    void onConnectBtnClicked() {
        if (uartConnected_) {
            drone_->disconnectUART();
            uartStatusLabel_->setText("DECONNECTE");
            uartStatusLabel_->setStyleSheet("color:#546e7a; font-weight:bold;");
            connectBtn_->setText("Connecter");
            uartConnected_ = false;
        } else {
            int baud = baudBox_->currentData().toInt();
            bool ok  = drone_->connectUART(
                serialPortBox_->currentText().toStdString(), baud);
            uartStatusLabel_->setText(ok ? "CONNECTE" : "ERREUR");
            uartStatusLabel_->setStyleSheet(ok ?
                "color:#00e676; font-weight:bold;" :
                "color:#ff1744; font-weight:bold;");
            connectBtn_->setText(ok ? "Deconnecter" : "Connecter");
            uartConnected_ = ok;
        }
    }

    void onRefreshPorts() {
        serialPortBox_->clear();
        auto ports = scanSerialPorts();
        if (ports.empty()) {
            serialPortBox_->addItem("/dev/ttyUSB0");
            serialPortBox_->addItem("/dev/ttyACM0");
        } else {
            for (auto& p : ports)
                serialPortBox_->addItem(QString::fromStdString(p));
        }
        log("[SYS] " + std::to_string(ports.size()) + " port(s) serie detecte(s)");
    }

    // ── Caméra ───────────────────────────────────────────────────────────────
    void onCamBtnClicked() {
        if (camConnected_) {
            drone_->closeCamera();
            camStatusLabel_->setText("INACTIVE");
            camStatusLabel_->setStyleSheet("color:#546e7a; font-weight:bold;");
            openCamBtn_->setText("Ouvrir camera");
            videoWidget_->setPixmap(QPixmap());
            videoWidget_->showNoSignal();
            camConnected_ = false;
        } else {
            bool ok = drone_->openCamera(
                camIndexSpin_->value(), 640, 480, fpsSpin_->value());
            camStatusLabel_->setText(ok ? "ACTIVE" : "ERREUR");
            camStatusLabel_->setStyleSheet(ok ?
                "color:#00e676; font-weight:bold;" :
                "color:#ff1744; font-weight:bold;");
            openCamBtn_->setText(ok ? "Fermer camera" : "Ouvrir camera");
            camConnected_ = ok;
        }
    }

    // ── IA ───────────────────────────────────────────────────────────────────
    void onStartAIBtnClicked() {
        if (aiRunning_) {
            drone_->stop();
            aiRunning_ = false;
            startAIBtn_->setText(">> Start IA");
            aiStatusLabel_->setText("IA ARRETEE");
            aiStatusLabel_->setStyleSheet("color:#546e7a; font-weight:bold;");
        } else {
            drone_->setThreshold((float)threshSlider_->value() / 100.f);
            drone_->setCRSFFreq(crsfHzSpin_->value());
            drone_->startAI();
            aiRunning_ = true;
            startAIBtn_->setText("[] Stop IA");
            aiStatusLabel_->setText("IA ACTIVE");
            aiStatusLabel_->setStyleSheet("color:#ffea00; font-weight:bold;");
        }
    }

    // ── Arming (avec popup de sécurité) ──────────────────────────────────────
    void onArmedChanged(int state) {
        bool armed = (state == Qt::Checked);
        if (armed) {
            auto ret = QMessageBox::warning(this, "SECURITE — ARMING",
                "ATTENTION : Vous allez ARMER le drone.\n\n"
                "Retirez les helices OU assurez-vous\n"
                "d'etre dans une zone completement securisee.\n\n"
                "Continuer ?",
                QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
            if (ret != QMessageBox::Yes) {
                armedCheck_->blockSignals(true);
                armedCheck_->setChecked(false);
                armedCheck_->blockSignals(false);
                return;
            }
        }
        drone_->setArmed(armed);
        armedLabel_->setText(armed ? "ARME" : "DESARME");
        armedLabel_->setStyleSheet(armed ?
            "color:#ff1744; font-size:13px; font-weight:bold;" :
            "color:#00e676; font-size:13px; font-weight:bold;");
        log(armed ? "[ARM] Drone ARME — CH5=2000" : "[ARM] Drone desarme — CH5=1000");
    }

    // ── Flush log vers QTextEdit ──────────────────────────────────────────────
    void flushLog() {
        QMutexLocker lock(&logMutex_);
        if (!pendingLogs_.isEmpty()) {
            logEdit_->append(pendingLogs_.join("\n"));
            pendingLogs_.clear();
            logEdit_->verticalScrollBar()->setValue(
                logEdit_->verticalScrollBar()->maximum());
        }
    }

private:
    // ── Construction de l'UI ─────────────────────────────────────────────────
    void buildUI() {
        auto* central    = new QWidget(this);
        auto* mainLayout = new QHBoxLayout(central);
        mainLayout->setContentsMargins(10, 10, 10, 10);
        mainLayout->setSpacing(10);
        setCentralWidget(central);

        // ────────── PANNEAU GAUCHE ──────────────────────────────────────────
        auto* leftPanel  = new QWidget;
        leftPanel->setMaximumWidth(320); leftPanel->setMinimumWidth(300);
        auto* leftLayout = new QVBoxLayout(leftPanel);
        leftLayout->setSpacing(8);

        auto* titleLbl = new QLabel("FPV·AI");
        titleLbl->setAlignment(Qt::AlignCenter);
        titleLbl->setStyleSheet(
            "font-size:26px; font-family:'Courier New'; font-weight:bold;"
            "color:#00b4ff; letter-spacing:5px;"
            "border-bottom:2px solid #0d3a6b; padding-bottom:6px;");
        leftLayout->addWidget(titleLbl);

        auto* subLbl = new QLabel("DRONE AI CONTROLLER — Pop!_OS");
        subLbl->setAlignment(Qt::AlignCenter);
        subLbl->setStyleSheet("font-size:8px; color:#1a3a6b; letter-spacing:2px;");
        leftLayout->addWidget(subLbl);
        leftLayout->addSpacing(4);

        // ── Groupe UART ──────────────────────────────────────────────────────
        auto* uartGroup  = buildGroup("LIAISON UART / ELRS");
        auto* uartLayout = new QGridLayout;
        uartLayout->setSpacing(5);

        uartLayout->addWidget(makeLabel("Port serie :"), 0, 0);
        serialPortBox_ = new QComboBox;
        serialPortBox_->setStyleSheet(comboStyle());
        // Remplir avec les ports disponibles
        auto ports = scanSerialPorts();
        if (ports.empty()) {
            serialPortBox_->addItem("/dev/ttyUSB0");
            serialPortBox_->addItem("/dev/ttyACM0");
        } else {
            for (auto& p : ports)
                serialPortBox_->addItem(QString::fromStdString(p));
        }
        uartLayout->addWidget(serialPortBox_, 0, 1);

        auto* refreshBtn = makeButton("Scan", "#112233");
        refreshBtn->setMaximumWidth(44);
        connect(refreshBtn, &QPushButton::clicked, this, &MainWindow::onRefreshPorts);
        uartLayout->addWidget(refreshBtn, 0, 2);

        uartLayout->addWidget(makeLabel("Baud :"), 1, 0);
        baudBox_ = new QComboBox;
        baudBox_->addItem("115200", 115200);
        baudBox_->addItem("230400", 230400);
        baudBox_->addItem("420000", 420000);
        baudBox_->addItem("460800", 460800);
        baudBox_->addItem("921600", 921600);
        baudBox_->setCurrentIndex(2); // 420000 par défaut
        baudBox_->setStyleSheet(comboStyle());
        uartLayout->addWidget(baudBox_, 1, 1, 1, 2);

        uartLayout->addWidget(makeLabel("Statut :"), 2, 0);
        uartStatusLabel_ = new QLabel("DECONNECTE");
        uartStatusLabel_->setStyleSheet("color:#546e7a; font-weight:bold; font-size:10px;");
        uartLayout->addWidget(uartStatusLabel_, 2, 1, 1, 2);

        connectBtn_ = makeButton("Connecter", "#063366");
        connect(connectBtn_, &QPushButton::clicked, this, &MainWindow::onConnectBtnClicked);
        uartLayout->addWidget(connectBtn_, 3, 0, 1, 3);

        uartGroup->layout()->addItem(uartLayout);
        leftLayout->addWidget(uartGroup);

        // ── Groupe Caméra ─────────────────────────────────────────────────────
        auto* camGroup  = buildGroup("CAMERA FPV (V4L2)");
        auto* camLayout = new QGridLayout;
        camLayout->setSpacing(5);

        camLayout->addWidget(makeLabel("/dev/video :"), 0, 0);
        camIndexSpin_ = new QSpinBox;
        camIndexSpin_->setRange(0, 9); camIndexSpin_->setValue(0);
        camIndexSpin_->setStyleSheet(spinStyle());
        camLayout->addWidget(camIndexSpin_, 0, 1);

        camLayout->addWidget(makeLabel("FPS cible :"), 1, 0);
        fpsSpin_ = new QSpinBox;
        fpsSpin_->setRange(15, 120); fpsSpin_->setValue(60);
        fpsSpin_->setStyleSheet(spinStyle());
        camLayout->addWidget(fpsSpin_, 1, 1);

        camLayout->addWidget(makeLabel("Statut :"), 2, 0);
        camStatusLabel_ = new QLabel("INACTIVE");
        camStatusLabel_->setStyleSheet("color:#546e7a; font-weight:bold; font-size:10px;");
        camLayout->addWidget(camStatusLabel_, 2, 1);

        openCamBtn_ = makeButton("Ouvrir camera", "#004422");
        connect(openCamBtn_, &QPushButton::clicked, this, &MainWindow::onCamBtnClicked);
        camLayout->addWidget(openCamBtn_, 3, 0, 1, 2);

        camGroup->layout()->addItem(camLayout);
        leftLayout->addWidget(camGroup);

        // ── Groupe IA ─────────────────────────────────────────────────────────
        auto* aiGroup  = buildGroup("INTELLIGENCE ARTIFICIELLE");
        auto* aiLayout = new QGridLayout;
        aiLayout->setSpacing(5);

        aiLayout->addWidget(makeLabel("Seuil obstacle :"), 0, 0);
        threshSlider_ = new QSlider(Qt::Horizontal);
        threshSlider_->setRange(10, 80); threshSlider_->setValue(35);
        threshSlider_->setStyleSheet(sliderStyle());
        aiLayout->addWidget(threshSlider_, 0, 1);
        threshValLabel_ = new QLabel("0.35");
        threshValLabel_->setStyleSheet("color:#00b4ff; font-family:'Courier New'; font-size:10px;");
        aiLayout->addWidget(threshValLabel_, 0, 2);
        connect(threshSlider_, &QSlider::valueChanged, [this](int v) {
            threshValLabel_->setText(QString::number(v / 100.f, 'f', 2));
            if (drone_) drone_->setThreshold(v / 100.f);
        });

        aiLayout->addWidget(makeLabel("Freq CRSF (Hz) :"), 1, 0);
        crsfHzSpin_ = new QSpinBox;
        crsfHzSpin_->setRange(50, 500); crsfHzSpin_->setValue(250);
        crsfHzSpin_->setStyleSheet(spinStyle());
        aiLayout->addWidget(crsfHzSpin_, 1, 1, 1, 2);

        aiLayout->addWidget(makeLabel("Statut :"), 2, 0);
        aiStatusLabel_ = new QLabel("IA ARRETEE");
        aiStatusLabel_->setStyleSheet("color:#546e7a; font-weight:bold; font-size:10px;");
        aiLayout->addWidget(aiStatusLabel_, 2, 1, 1, 2);

        startAIBtn_ = new QPushButton(">> Start IA");
        startAIBtn_->setMinimumHeight(38);
        startAIBtn_->setStyleSheet(
            "QPushButton { background:#002800; color:#00ff88;"
            " border:2px solid #00aa44; border-radius:6px;"
            " font-size:13px; font-weight:bold; font-family:'Courier New'; }"
            "QPushButton:hover { background:#004400; }"
            "QPushButton:pressed { background:#001800; }");
        connect(startAIBtn_, &QPushButton::clicked, this, &MainWindow::onStartAIBtnClicked);
        aiLayout->addWidget(startAIBtn_, 3, 0, 1, 3);

        aiGroup->layout()->addItem(aiLayout);
        leftLayout->addWidget(aiGroup);

        // ── Arming ────────────────────────────────────────────────────────────
        auto* armGroup  = buildGroup("ARMING (SECURITE)");
        auto* armLayout = new QHBoxLayout;
        armedCheck_ = new QCheckBox("Activer arming CH5");
        armedCheck_->setStyleSheet("color:#ff5252; font-weight:bold; font-size:10px;");
        connect(armedCheck_, &QCheckBox::stateChanged, this, &MainWindow::onArmedChanged);
        armLayout->addWidget(armedCheck_);
        armedLabel_ = new QLabel("DESARME");
        armedLabel_->setStyleSheet("color:#00e676; font-size:13px; font-weight:bold;");
        armLayout->addWidget(armedLabel_);
        armGroup->layout()->addItem(armLayout);
        leftLayout->addWidget(armGroup);

        // ── Canaux RC ─────────────────────────────────────────────────────────
        auto* rcGroup  = buildGroup("CANAUX RC");
        auto* rcLayout = new QVBoxLayout;
        rcLayout->setSpacing(3);
        const char* chNames[] = {"CH1 Roll ","CH2 Pitch","CH3 Throt","CH4 Yaw  ","CH5 Arm  "};
        for (int i = 0; i < 5; i++) {
            rcBars_[i] = new RCBar(chNames[i]);
            rcLayout->addWidget(rcBars_[i]);
        }
        rcGroup->layout()->addItem(rcLayout);
        leftLayout->addWidget(rcGroup);

        leftLayout->addStretch();
        mainLayout->addWidget(leftPanel);

        // ────────── PANNEAU CENTRAL ─────────────────────────────────────────
        auto* centerPanel  = new QWidget;
        auto* centerLayout = new QVBoxLayout(centerPanel);
        centerLayout->setSpacing(8);

        // Barre de statut haut
        auto* statusBar    = new QWidget;
        statusBar->setFixedHeight(34);
        statusBar->setStyleSheet("background:#050810; border-bottom:1px solid #0d1f3c;");
        auto* statusLayout = new QHBoxLayout(statusBar);
        statusLayout->setContentsMargins(10, 0, 10, 0);

        latencyLabel_ = new QLabel("Latence IA : -- ms");
        latencyLabel_->setStyleSheet(
            "color:#00b4ff; font-family:'Courier New'; font-size:11px;");
        statusLayout->addWidget(latencyLabel_);
        statusLayout->addStretch();

        failsafeLabel_ = new QLabel("FAILSAFE : OK");
        failsafeLabel_->setStyleSheet(
            "color:#00e676; font-family:'Courier New'; font-size:11px;");
        statusLayout->addWidget(failsafeLabel_);

        auto* clockLbl = new QLabel;
        clockLbl->setStyleSheet("color:#2a4a7f; font-family:'Courier New'; font-size:10px;");
        statusLayout->addWidget(clockLbl);
        auto* clockTmr = new QTimer(this);
        connect(clockTmr, &QTimer::timeout, [clockLbl] {
            clockLbl->setText(QDateTime::currentDateTime().toString("hh:mm:ss"));
        });
        clockTmr->start(1000);

        centerLayout->addWidget(statusBar);

        // Widget vidéo
        videoWidget_ = new VideoWidget;
        videoWidget_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        centerLayout->addWidget(videoWidget_, 3);

        // Console log
        auto* logGroup = buildGroup("CONSOLE");
        logEdit_ = new QTextEdit;
        logEdit_->setReadOnly(true);
        logEdit_->setMaximumHeight(160);
        logEdit_->setFont(QFont("Courier New", 9));
        logEdit_->setStyleSheet(
            "background:#030608; color:#2a7fff; border:none;"
            " selection-background-color:#0d2244;");
        logEdit_->setPlainText(
            "[INIT] FPV Drone AI Controller — Pop!_OS / Linux\n"
            "[INIT] Connecter le module ELRS (ttyUSB0 ou ttyACM0)\n"
            "[INIT] Puis ouvrir /dev/video0 pour la camera\n"
            "[INIT] !! TESTER SANS HELICES EN PREMIER !!\n"
            "[INFO] Pour les priorites RT : sudo ./fpv_drone_ai\n"
            "[INFO] Ou : sudo setcap cap_sys_nice+ep ./fpv_drone_ai\n");

        auto* clearBtn = makeButton("Vider", "#111");
        clearBtn->setMaximumWidth(56); clearBtn->setMaximumHeight(22);
        connect(clearBtn, &QPushButton::clicked, logEdit_, &QTextEdit::clear);

        auto* logInner  = new QVBoxLayout;
        auto* logHeader = new QHBoxLayout;
        logHeader->addWidget(makeLabel("LOG"), 1);
        logHeader->addWidget(clearBtn);
        logInner->addItem(logHeader);
        logInner->addWidget(logEdit_);
        logGroup->layout()->addItem(logInner);
        centerLayout->addWidget(logGroup, 1);

        mainLayout->addWidget(centerPanel, 1);

        // ── Initialisation du DroneController ────────────────────────────────
        UiBridge bridge;
        bridge.onFrame = [this](const cv::Mat& img, const FlightCommand& cmd,
                                 float dL, float dC, float dR) {
            cv::Mat rgb;
            cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
            QImage qimg(rgb.data, rgb.cols, rgb.rows,
                        (int)rgb.step, QImage::Format_RGB888);
            qimg = qimg.copy();
            QMetaObject::invokeMethod(this, [this, qimg, cmd, dL, dC, dR]() {
                videoWidget_->updateFrame(qimg, cmd, dL, dC, dR);
            }, Qt::QueuedConnection);
        };

        bridge.onLog = [this](const std::string& msg) {
            QMutexLocker lock(&logMutex_);
            pendingLogs_ << QString("[%1] %2")
                .arg(QDateTime::currentDateTime().toString("hh:mm:ss.zzz"))
                .arg(QString::fromStdString(msg));
        };

        bridge.onLatency = [this](long ms) {
            QMetaObject::invokeMethod(this, [this, ms]() {
                QString sty = ms > 30 ? "color:#ff5252;" : "color:#00b4ff;";
                latencyLabel_->setText(QString("Latence IA : %1 ms").arg(ms));
                latencyLabel_->setStyleSheet(
                    sty + " font-family:'Courier New'; font-size:11px;");
            }, Qt::QueuedConnection);
        };

        bridge.onChannels = [this](const RCChannels& ch) {
            QMetaObject::invokeMethod(this, [this, ch]() {
                for (int i = 0; i < 5; i++) rcBars_[i]->setValue(ch[i]);
            }, Qt::QueuedConnection);
        };

        bridge.onFailsafe = [this](bool active) {
            QMetaObject::invokeMethod(this, [this, active]() {
                failsafeLabel_->setText(active ? "FAILSAFE ACTIF!" : "FAILSAFE : OK");
                failsafeLabel_->setStyleSheet(active ?
                    "color:#ff1744; font-weight:bold; font-family:'Courier New';" :
                    "color:#00e676; font-family:'Courier New'; font-size:11px;");
            }, Qt::QueuedConnection);
        };

        drone_ = std::make_unique<DroneController>(std::move(bridge));
        log("[INIT] Controller initialise");
    }

    // ── Thème sombre ──────────────────────────────────────────────────────────
    void applyDarkTheme() {
        QPalette pal;
        pal.setColor(QPalette::Window,          QColor(6, 10, 22));
        pal.setColor(QPalette::WindowText,      QColor(180, 200, 240));
        pal.setColor(QPalette::Base,            QColor(4, 8, 18));
        pal.setColor(QPalette::AlternateBase,   QColor(8, 14, 30));
        pal.setColor(QPalette::Text,            QColor(140, 180, 240));
        pal.setColor(QPalette::Button,          QColor(10, 20, 44));
        pal.setColor(QPalette::ButtonText,      QColor(140, 180, 240));
        pal.setColor(QPalette::Highlight,       QColor(0, 80, 180));
        pal.setColor(QPalette::HighlightedText, Qt::white);
        qApp->setPalette(pal);
        qApp->setStyle(QStyleFactory::create("Fusion"));
    }

    // ── Helpers constructeurs de widgets ──────────────────────────────────────
    QGroupBox* buildGroup(const QString& title) {
        auto* g = new QGroupBox(title);
        g->setStyleSheet(
            "QGroupBox { border:1px solid #0d2a55; border-radius:6px;"
            " margin-top:10px; color:#2a6bbb;"
            " font-family:'Courier New'; font-size:10px; }"
            "QGroupBox::title { subcontrol-origin:margin; left:8px; padding:0 4px; }");
        auto* lay = new QVBoxLayout(g);
        lay->setContentsMargins(8, 12, 8, 8);
        lay->setSpacing(4);
        return g;
    }

    QLabel* makeLabel(const QString& txt) {
        auto* l = new QLabel(txt);
        l->setStyleSheet("color:#4a7ab0; font-family:'Courier New'; font-size:10px;");
        return l;
    }

    QPushButton* makeButton(const QString& txt, const QString& bg) {
        auto* btn = new QPushButton(txt);
        btn->setMinimumHeight(26);
        btn->setStyleSheet(QString(
            "QPushButton { background:%1; color:#aaccff;"
            " border:1px solid #1a3a6b; border-radius:4px;"
            " font-family:'Courier New'; font-size:10px; }"
            "QPushButton:hover { background:#1a3a6b; }"
            "QPushButton:pressed { background:#0a1a3a; }").arg(bg));
        return btn;
    }

    QString comboStyle() {
        return "QComboBox { background:#04080e; color:#88aadd;"
               " border:1px solid #0d2244; border-radius:3px;"
               " font-family:'Courier New'; font-size:10px; padding:2px 6px; }"
               "QComboBox::drop-down { border:none; }"
               "QComboBox QAbstractItemView { background:#04080e; color:#88aadd; }";
    }

    QString spinStyle() {
        return "QSpinBox { background:#04080e; color:#88aadd;"
               " border:1px solid #0d2244; border-radius:3px;"
               " font-family:'Courier New'; font-size:10px; }";
    }

    QString sliderStyle() {
        return "QSlider::groove:horizontal { background:#0a1a3a; height:5px; border-radius:2px; }"
               "QSlider::handle:horizontal { background:#0066cc; width:13px; height:13px;"
               " margin:-4px 0; border-radius:7px; }"
               "QSlider::sub-page:horizontal { background:#003388; border-radius:2px; }";
    }

    void log(const std::string& msg) {
        QMutexLocker lock(&logMutex_);
        pendingLogs_ << QString("[%1] %2")
            .arg(QDateTime::currentDateTime().toString("hh:mm:ss.zzz"))
            .arg(QString::fromStdString(msg));
    }

    void closeEvent(QCloseEvent* e) override {
        if (drone_) drone_->stop();
        QMainWindow::closeEvent(e);
    }

    // Widgets
    VideoWidget*  videoWidget_     = nullptr;
    QComboBox*    serialPortBox_   = nullptr;
    QComboBox*    baudBox_         = nullptr;
    QPushButton*  connectBtn_      = nullptr;
    QPushButton*  refreshPortsBtn_ = nullptr;
    QLabel*       uartStatusLabel_ = nullptr;
    QSpinBox*     camIndexSpin_    = nullptr;
    QSpinBox*     fpsSpin_         = nullptr;
    QPushButton*  openCamBtn_      = nullptr;
    QLabel*       camStatusLabel_  = nullptr;
    QSlider*      threshSlider_    = nullptr;
    QLabel*       threshValLabel_  = nullptr;
    QSpinBox*     crsfHzSpin_      = nullptr;
    QPushButton*  startAIBtn_      = nullptr;
    QLabel*       aiStatusLabel_   = nullptr;
    QCheckBox*    armedCheck_      = nullptr;
    QLabel*       armedLabel_      = nullptr;
    RCBar*        rcBars_[5]       = {};
    QTextEdit*    logEdit_         = nullptr;
    QLabel*       latencyLabel_    = nullptr;
    QLabel*       failsafeLabel_   = nullptr;

    // Logique
    std::unique_ptr<DroneController> drone_;
    bool uartConnected_ = false;
    bool camConnected_  = false;
    bool aiRunning_     = false;

    // Log thread-safe
    QMutex      logMutex_;
    QStringList pendingLogs_;
};

// ─────────────────────────────────────────────────────────────────────────────
//  MOC inline (évite un fichier .moc séparé avec CMake AUTOMOC)
// ─────────────────────────────────────────────────────────────────────────────
#include "fpv_drone_gui_linux.moc"

// ─────────────────────────────────────────────────────────────────────────────
//  MAIN
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    QApplication app(argc, argv);
    app.setApplicationName("FPV Drone AI — Pop!_OS");
    app.setApplicationVersion("1.0-linux");

    MainWindow w;
    w.show();
    return app.exec();
}
