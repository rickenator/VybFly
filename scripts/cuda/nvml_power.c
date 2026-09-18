// Minimal NVML power probe: prints "power_mW util_pct sm_mhz" every interval, so the
// harness can compare NVML's own counter against `nvidia-smi --query-gpu=power.draw`.
// Build: gcc -O2 -o nvml_power nvml_power.c -lnvidia-ml
#include <nvml.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char** argv) {
  int samples = argc > 1 ? atoi(argv[1]) : 10;
  int interval_ms = argc > 2 ? atoi(argv[2]) : 500;
  if (nvmlInit_v2() != NVML_SUCCESS) { fprintf(stderr, "nvmlInit failed\n"); return 1; }
  nvmlDevice_t dev;
  if (nvmlDeviceGetHandleByIndex_v2(0, &dev) != NVML_SUCCESS) {
    fprintf(stderr, "no device 0\n"); return 1;
  }
  for (int i = 0; i < samples; ++i) {
    unsigned int mw = 0, util = 0, sm = 0;
    unsigned int power_limit = 0;
    nvmlReturn_t r = nvmlDeviceGetPowerUsage(dev, &mw);
    nvmlDeviceGetUtilizationRates(dev, &(nvmlUtilization_t){0});
    nvmlUtilization_t u = {0, 0};
    nvmlDeviceGetUtilizationRates(dev, &u);
    nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &sm);
    nvmlDeviceGetEnforcedPowerLimit(dev, &power_limit);
    printf("nvml power_mw=%u power_w=%.3f util_gpu_pct=%u sm_mhz=%u limit_mw=%u rc=%d\n",
           mw, mw / 1000.0, u.gpu, sm, power_limit, (int)r);
    fflush(stdout);
    usleep(interval_ms * 1000);
  }
  nvmlShutdown();
  return 0;
}
