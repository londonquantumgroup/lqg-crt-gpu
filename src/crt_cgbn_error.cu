#include "../include/crt_cgbn.hpp"

#ifdef ENABLE_CGBN

// Implement error report functions once here
void cgbn_error_report_alloc_impl(cgbn_error_report_t **report) {
    if (report) {
        cudaMallocManaged(report, sizeof(cgbn_error_report_t));
        if (*report) {
            memset(*report, 0, sizeof(cgbn_error_report_t));
        }
    }
}

void cgbn_error_report_free_impl(cgbn_error_report_t *report) {
    if (report) {
        cudaFree(report);
    }
}

void cgbn_error_report_check_impl(cgbn_error_report_t *report) {
    // Check for errors in the report
    if (report && report->_instance != 0xFFFFFFFF) {
        fprintf(stderr, "CGBN error at instance %d\n", report->_instance);
    }
}

void cgbn_error_report_reset_impl(cgbn_error_report_t *report) {
    if (report) {
        memset(report, 0, sizeof(cgbn_error_report_t));
    }
}

const char* cgbn_error_string_impl(cgbn_error_report_t *report) {
    if (!report) return "No report";
    if (report->_instance == 0xFFFFFFFF) return "No error";
    
    static char buffer[256];
    snprintf(buffer, sizeof(buffer), 
             "Error at instance %d, error code %d", 
             report->_instance, report->_error);
    return buffer;
}

#endif // ENABLE_CGBN