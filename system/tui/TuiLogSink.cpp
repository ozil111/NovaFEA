/**
 * TUI Log Sink - spdlog sink that mirrors log output into the TUI output pane.
 */
#include "tui/AppTuiState.h"
#include <spdlog/spdlog.h>
#include "spdlog/sinks/base_sink.h"
#include <ftxui/dom/elements.hpp>
#include <fmt/format.h>
#include <algorithm>
#include <deque>
#include <mutex>

namespace tui {
using namespace ftxui;

namespace {

std::mutex g_tui_log_mutex;
std::deque<TuiLogLine> g_tui_log_lines;
std::shared_ptr<spdlog::sinks::sink> g_tui_log_sink;

class TuiLogSink final : public spdlog::sinks::base_sink<std::mutex> {
protected:
    void sink_it_(const spdlog::details::log_msg& msg) override {
        spdlog::memory_buf_t formatted;
        formatter_->format(msg, formatted);
        std::string line = fmt::to_string(formatted);
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
            line.pop_back();
        }
        if (line.empty()) return;

        std::lock_guard<std::mutex> lock(g_tui_log_mutex);
        g_tui_log_lines.push_back(TuiLogLine{ msg.level, std::move(line) });
        if (g_tui_log_lines.size() > 1000) {
            g_tui_log_lines.erase(g_tui_log_lines.begin(), g_tui_log_lines.begin() + 200);
        }
    }

    void flush_() override {}
};

} // namespace

// ── Utility functions ─────────────────────────────────────────────────

float clamp01(float v) {
    if (v < 0.0f) return 0.0f;
    if (v > 1.0f) return 1.0f;
    return v;
}

bool starts_with(const std::string& s, const std::string& prefix) {
    return s.size() >= prefix.size() && std::equal(prefix.begin(), prefix.end(), s.begin());
}

// ── Log rendering helpers ─────────────────────────────────────────────

Decorator level_decorator(spdlog::level::level_enum lvl) {
    using spdlog::level::level_enum;
    switch (lvl) {
        case level_enum::trace:    return dim;
        case level_enum::debug:    return color(Color::GrayDark);
        case level_enum::info:     return color(Color::GreenLight);
        case level_enum::warn:     return color(Color::YellowLight);
        case level_enum::err:      return color(Color::RedLight);
        case level_enum::critical: return color(Color::RedLight) | bold;
        default:                   return nothing;
    }
}

Element status_lines_element(const std::vector<std::string>& lines) {
    Elements els;
    els.reserve(lines.size());
    for (const auto& s : lines) {
        els.push_back(paragraph(s));
    }
    return vbox(std::move(els));
}

Element status_lines_element(const std::vector<TuiLogLine>& lines) {
    Elements els;
    els.reserve(lines.size());
    for (const auto& s : lines) {
        els.push_back(paragraph(s.text) | level_decorator(s.level));
    }
    return vbox(std::move(els));
}

// ── Log snapshot ──────────────────────────────────────────────────────

std::vector<TuiLogLine> tui_log_lines_snapshot() {
    std::lock_guard<std::mutex> lock(g_tui_log_mutex);
    return std::vector<TuiLogLine>(g_tui_log_lines.begin(), g_tui_log_lines.end());
}

// ── Sink installation ─────────────────────────────────────────────────

void install_tui_log_sink() {
    if (g_tui_log_sink) return;
    auto logger = spdlog::default_logger();
    if (!logger) return;

    auto sink = std::make_shared<TuiLogSink>();
    logger->sinks().push_back(sink);
    g_tui_log_sink = std::move(sink);
}

} // namespace tui
