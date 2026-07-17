#pragma once

#include <string>
#include <string.h>
#include <exception>
#include <sstream>
#include <cstdint>

namespace rtp_llm {

enum class CpKvLayoutKind : uint8_t {
    REPLICATED                = 0,
    LEGACY_DYNAMIC_CONTIGUOUS = 1,
    PAGE_INTERLEAVED          = 2,
};

struct CpKvLayoutConfig {
    CpKvLayoutKind kind              = CpKvLayoutKind::REPLICATED;
    int            cp_size           = 1;
    int            cp_rank           = 0;
    int            page_size         = 0;
    int            interleave_size   = 0;
    uint32_t       layout_version    = 0;
    int64_t        max_global_length = 0;
};

enum class OpErrorType {
    ERROR_NONE,
    ERROR_INVALID_ARGS,
    ERROR_RESOURCE_EXHAUSTED,
    ERROR_UNIMPLEMENTED,
    ERROR_INTERNAL,
    ERROR_UNKNOWN,
};

class OpStatus {
public:
    OpStatus(OpErrorType error_type, const std::string& message = ""): error_type(error_type), error_message(message) {}

    static OpStatus make(OpErrorType error_type, const std::string& error_message = "") {
        return OpStatus(error_type, error_message);
    }
    static OpStatus OK() {
        return OpStatus(OpErrorType::ERROR_NONE);
    }

    bool ok() const {
        return error_type == OpErrorType::ERROR_NONE;
    }

public:
    OpErrorType error_type;
    std::string error_message;
};

class OpException: public std::exception {
public:
    OpException(const OpStatus& status);

    const char* what() const noexcept override {
        return detail_str_.c_str();
    }

    const OpStatus& status() const {
        return status_;
    }

private:
    OpStatus            status_;
    mutable std::string detail_str_;
};

}  // namespace rtp_llm
