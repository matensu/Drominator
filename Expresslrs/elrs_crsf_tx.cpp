// elrs_crsf_tx.cpp
#include <array>
#include <chrono>
#include <cstdint>
#include <exception>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#endif

namespace crsf {

constexpr uint8_t SYNC_BYTE = 0xC8;
constexpr uint8_t FRAME_TYPE_RC_CHANNELS_PACKED = 0x16;
constexpr int CHANNEL_COUNT = 16;
constexpr int PAYLOAD_SIZE = 22;
constexpr int FRAME_SIZE = 26; // sync + len + type + payload + crc

constexpr uint16_t TICKS_MIN = 172;
constexpr uint16_t TICKS_MID = 992;
constexpr uint16_t TICKS_MAX = 1811;

int clamp_us(int value) {
    if (value < 988) return 988;
    if (value > 2012) return 2012;
    return value;
}

uint16_t us_to_ticks(int us) {
    us = clamp_us(us);
    int ticks = ((us - 1500) * 8) / 5 + 992;
    if (ticks < TICKS_MIN) ticks = TICKS_MIN;
    if (ticks > TICKS_MAX) ticks = TICKS_MAX;
    return static_cast<uint16_t>(ticks);
}

uint8_t crc8_d5(const uint8_t* data, size_t length) {
    uint8_t crc = 0;
    for (size_t i = 0; i < length; ++i) {
        crc ^= data[i];
        for (int bit = 0; bit < 8; ++bit) {
            if (crc & 0x80) {
                crc = static_cast<uint8_t>((crc << 1) ^ 0xD5);
            } else {
                crc = static_cast<uint8_t>(crc << 1);
            }
        }
    }
    return crc;
}

std::array<uint8_t, FRAME_SIZE> build_rc_frame(const std::array<int, CHANNEL_COUNT>& channels_us) {
    std::array<uint8_t, FRAME_SIZE> frame{};
    std::array<uint16_t, CHANNEL_COUNT> channels_ticks{};

    for (int i = 0; i < CHANNEL_COUNT; ++i) {
        channels_ticks[i] = us_to_ticks(channels_us[i]);
    }

    frame[0] = SYNC_BYTE;
    frame[1] = 24; // type + payload + crc = 1 + 22 + 1
    frame[2] = FRAME_TYPE_RC_CHANNELS_PACKED;

    uint32_t bit_buffer = 0;
    int bit_count = 0;
    int out_index = 3;

    for (int i = 0; i < CHANNEL_COUNT; ++i) {
        bit_buffer |= (static_cast<uint32_t>(channels_ticks[i] & 0x07FF) << bit_count);
        bit_count += 11;

        while (bit_count >= 8) {
            frame[out_index++] = static_cast<uint8_t>(bit_buffer & 0xFF);
            bit_buffer >>= 8;
            bit_count -= 8;
        }
    }

    frame[25] = crc8_d5(&frame[2], 23); // type + payload
    return frame;
}

} // namespace crsf

class SerialPort {
public:
    SerialPort() = default;
    ~SerialPort() { close(); }

    void open(const std::string& port_name, int baud_rate) {
#ifdef _WIN32
        handle_ = CreateFileA(
            port_name.c_str(),
            GENERIC_READ | GENERIC_WRITE,
            0,
            nullptr,
            OPEN_EXISTING,
            0,
            nullptr
        );

        if (handle_ == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("Impossible d'ouvrir le port série: " + port_name);
        }

        DCB dcb{};
        dcb.DCBlength = sizeof(dcb);

        if (!GetCommState(handle_, &dcb)) {
            close();
            throw std::runtime_error("GetCommState a échoué");
        }

        dcb.BaudRate = baud_rate;
        dcb.ByteSize = 8;
        dcb.Parity = NOPARITY;
        dcb.StopBits = ONESTOPBIT;
        dcb.fBinary = TRUE;
        dcb.fOutxCtsFlow = FALSE;
        dcb.fOutxDsrFlow = FALSE;
        dcb.fDtrControl = DTR_CONTROL_DISABLE;
        dcb.fDsrSensitivity = FALSE;
        dcb.fRtsControl = RTS_CONTROL_DISABLE;
        dcb.fOutX = FALSE;
        dcb.fInX = FALSE;

        if (!SetCommState(handle_, &dcb)) {
            close();
            throw std::runtime_error("SetCommState a échoué");
        }

        COMMTIMEOUTS timeouts{};
        timeouts.ReadIntervalTimeout = 1;
        timeouts.ReadTotalTimeoutConstant = 1;
        timeouts.ReadTotalTimeoutMultiplier = 1;
        timeouts.WriteTotalTimeoutConstant = 1;
        timeouts.WriteTotalTimeoutMultiplier = 1;

        if (!SetCommTimeouts(handle_, &timeouts)) {
            close();
            throw std::runtime_error("SetCommTimeouts a échoué");
        }

#else
        fd_ = ::open(port_name.c_str(), O_RDWR | O_NOCTTY | O_SYNC);
        if (fd_ < 0) {
            throw std::runtime_error("Impossible d'ouvrir le port série: " + port_name +
                                     " errno=" + std::to_string(errno));
        }

        termios tty{};
        if (tcgetattr(fd_, &tty) != 0) {
            close();
            throw std::runtime_error("tcgetattr a échoué");
        }

        cfmakeraw(&tty);

        speed_t speed = baud_to_constant(baud_rate);
        cfsetispeed(&tty, speed);
        cfsetospeed(&tty, speed);

        tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
        tty.c_cflag |= (CLOCAL | CREAD);
        tty.c_cflag &= ~(PARENB | PARODD);
        tty.c_cflag &= ~CSTOPB;
        tty.c_cflag &= ~CRTSCTS;

        tty.c_cc[VMIN] = 0;
        tty.c_cc[VTIME] = 0;

        if (tcsetattr(fd_, TCSANOW, &tty) != 0) {
            close();
            throw std::runtime_error("tcsetattr a échoué");
        }
#endif
    }

    void write_all(const uint8_t* data, size_t length) {
#ifdef _WIN32
        DWORD total_written = 0;
        while (total_written < length) {
            DWORD written = 0;
            if (!WriteFile(handle_, data + total_written,
                           static_cast<DWORD>(length - total_written),
                           &written, nullptr)) {
                throw std::runtime_error("WriteFile a échoué");
            }
            total_written += written;
        }
#else
        size_t total_written = 0;
        while (total_written < length) {
            ssize_t written = ::write(fd_, data + total_written, length - total_written);
            if (written < 0) {
                if (errno == EINTR) continue;
                throw std::runtime_error("write a échoué errno=" + std::to_string(errno));
            }
            total_written += static_cast<size_t>(written);
        }
#endif
    }

    void close() {
#ifdef _WIN32
        if (handle_ != INVALID_HANDLE_VALUE) {
            CloseHandle(handle_);
            handle_ = INVALID_HANDLE_VALUE;
        }
#else
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
#endif
    }

private:
#ifndef _WIN32
    static speed_t baud_to_constant(int baud_rate) {
        switch (baud_rate) {
            case 115200: return B115200;
#ifdef B230400
            case 230400: return B230400;
#endif
#ifdef B460800
            case 460800: return B460800;
#endif
#ifdef B921600
            case 921600: return B921600;
#endif
            default:
                throw std::runtime_error("Baud rate non supporté sur cette plateforme: " +
                                         std::to_string(baud_rate));
        }
    }
#endif

#ifdef _WIN32
    HANDLE handle_ = INVALID_HANDLE_VALUE;
#else
    int fd_ = -1;
#endif
};

struct RcState {
    std::mutex mutex;
    std::array<int, 16> channels_us = {
        1500, // ch1 roll
        1500, // ch2 pitch
        1000, // ch3 throttle
        1500, // ch4 yaw
        1000, // ch5 arm
        1000, // ch6 mode
        1000, // ch7
        1000, // ch8
        1000, // ch9
        1000, // ch10
        1000, // ch11
        1000, // ch12
        1000, // ch13
        1000, // ch14
        1000, // ch15
        1000  // ch16
    };
    bool running = true;
};

void print_channels(const std::array<int, 16>& channels) {
    for (int i = 0; i < 16; ++i) {
        std::cout << "ch" << (i + 1) << "=" << channels[i];
        if (i != 15) std::cout << ' ';
    }
    std::cout << '\n';
}

void print_help() {
    std::cout
        << "Commandes disponibles:\n"
        << "  help                              affiche l'aide\n"
        << "  show                              affiche les 16 canaux\n"
        << "  set <canal> <valeur_us>           ex: set 3 1200\n"
        << "  sticks <roll> <pitch> <thr> <yaw> ex: sticks 1500 1500 1000 1500\n"
        << "  arm                               met ch5 a 2000\n"
        << "  disarm                            met ch5 a 1000\n"
        << "  mode1                             met ch6 a 1000\n"
        << "  mode2                             met ch6 a 1500\n"
        << "  mode3                             met ch6 a 2000\n"
        << "  center                            centre roll pitch yaw\n"
        << "  idle                              met throttle a 1000\n"
        << "  stop                              coupe throttle + disarm\n"
        << "  quit                              quitte le programme\n";
}

void tx_loop(SerialPort& serial, RcState& state, int rate_hz) {
    const auto period = std::chrono::microseconds(1000000 / rate_hz);

    while (true) {
        std::array<int, 16> snapshot{};
        {
            std::lock_guard<std::mutex> lock(state.mutex);
            if (!state.running) break;
            snapshot = state.channels_us;
        }

        const auto frame = crsf::build_rc_frame(snapshot);
        serial.write_all(frame.data(), frame.size());
        std::this_thread::sleep_for(period);
    }
}

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::cerr
                << "Usage:\n"
                << "  Windows: elrs_crsf_tx.exe COM3 [baud] [rate_hz]\n"
                << "  Linux:   ./elrs_crsf_tx /dev/ttyUSB0 [baud] [rate_hz]\n"
                << "\n"
                << "Exemples:\n"
                << "  elrs_crsf_tx.exe COM3 460800 100\n"
                << "  ./elrs_crsf_tx /dev/ttyUSB0 460800 100\n";
            return 1;
        }

        std::string port_name = argv[1];
#ifdef _WIN32
        if (port_name.rfind("\\\\.\\", 0) != 0 && port_name.rfind("COM", 0) == 0) {
            port_name = "\\\\.\\" + port_name;
        }
#endif

        const int baud_rate = (argc >= 3) ? std::stoi(argv[2]) : 460800;
        const int rate_hz   = (argc >= 4) ? std::stoi(argv[3]) : 100;

        SerialPort serial;
        serial.open(port_name, baud_rate);

        RcState state;
        std::thread sender(tx_loop, std::ref(serial), std::ref(state), rate_hz);

        std::cout << "Flux CRSF actif sur " << argv[1]
                  << " a " << baud_rate << " bauds, "
                  << rate_hz << " Hz\n";
        std::cout << "Retire les helices avant test.\n";
        print_help();

        std::string line;
        while (std::getline(std::cin, line)) {
            std::istringstream iss(line);
            std::string cmd;
            iss >> cmd;
            if (cmd.empty()) continue;

            std::lock_guard<std::mutex> lock(state.mutex);

            if (cmd == "help") {
                print_help();
            } else if (cmd == "show") {
                print_channels(state.channels_us);
            } else if (cmd == "set") {
                int channel = 0;
                int value = 0;
                if (!(iss >> channel >> value) || channel < 1 || channel > 16) {
                    std::cout << "Usage: set <1..16> <988..2012>\n";
                    continue;
                }
                state.channels_us[channel - 1] = crsf::clamp_us(value);
                print_channels(state.channels_us);
            } else if (cmd == "sticks") {
                int roll = 1500, pitch = 1500, thr = 1000, yaw = 1500;
                if (!(iss >> roll >> pitch >> thr >> yaw)) {
                    std::cout << "Usage: sticks <roll> <pitch> <thr> <yaw>\n";
                    continue;
                }
                state.channels_us[0] = crsf::clamp_us(roll);
                state.channels_us[1] = crsf::clamp_us(pitch);
                state.channels_us[2] = crsf::clamp_us(thr);
                state.channels_us[3] = crsf::clamp_us(yaw);
                print_channels(state.channels_us);
            } else if (cmd == "arm") {
                state.channels_us[4] = 2000;
                print_channels(state.channels_us);
            } else if (cmd == "disarm") {
                state.channels_us[4] = 1000;
                print_channels(state.channels_us);
            } else if (cmd == "mode1") {
                state.channels_us[5] = 1000;
                print_channels(state.channels_us);
            } else if (cmd == "mode2") {
                state.channels_us[5] = 1500;
                print_channels(state.channels_us);
            } else if (cmd == "mode3") {
                state.channels_us[5] = 2000;
                print_channels(state.channels_us);
            } else if (cmd == "center") {
                state.channels_us[0] = 1500;
                state.channels_us[1] = 1500;
                state.channels_us[3] = 1500;
                print_channels(state.channels_us);
            } else if (cmd == "idle") {
                state.channels_us[2] = 1000;
                print_channels(state.channels_us);
            } else if (cmd == "stop") {
                state.channels_us[2] = 1000;
                state.channels_us[4] = 1000;
                print_channels(state.channels_us);
            } else if (cmd == "quit" || cmd == "exit") {
                state.channels_us[2] = 1000;
                state.channels_us[4] = 1000;
                state.running = false;
                break;
            } else {
                std::cout << "Commande inconnue. Tape help\n";
            }
        }

        {
            std::lock_guard<std::mutex> lock(state.mutex);
            state.channels_us[2] = 1000;
            state.channels_us[4] = 1000;
            state.running = false;
        }

        if (sender.joinable()) {
            sender.join();
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Erreur: " << e.what() << '\n';
        return 1;
    }
}