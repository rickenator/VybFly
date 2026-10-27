// make_stim.cpp — write the ladder's stimulus schedule for one scale.
//
// The ladder harness (src/vyb/ladder_*.vyb) reads a binary schedule:
//
//     int32 n_steps
//     int32 offsets[n_steps + 1]     // event range per step
//     int32 neuron_ids[n_events]
//
// This is the same schedule scripts/cuda/sparse_prop.cu builds internally when it is
// run without --stim-file (mt19937_64, seed 1337, 500 stimulus neurons at 5 Hz, dt
// 0.5 ms). The ids are drawn from [0, n_neurons), so each scale needs its own file —
// which is why data/stim_s10.bin and data/stim_s80.bin differ while their offsets are
// identical. Reproducing that generator exactly (verified: byte-identical to
// stim_s1.bin at n=139255 and to stim_s80.bin at n=11140400) is what makes a new
// scale comparable to the published ones instead of merely similar.
//
// Build and run:
//     g++ -O2 -o make_stim scripts/make_stim.cpp
//     ./make_stim <n_neurons> <out.bin>            # 100x: n = 13925500
//
// Optional overrides, matching sparse_prop's flags:
//     ./make_stim <n> <out> <steps> <demo_hz> <dt_ms> <seed>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr,
            "usage: %s <n_neurons> <out.bin> [steps=1000] [stim_hz=5.0] [dt_ms=0.5] [seed=1337]\n",
            argv[0]);
    return 2;
  }
  const int n = atoi(argv[1]);
  const char* out = argv[2];
  const int steps = argc > 3 ? atoi(argv[3]) : 1000;
  const double stim_hz = argc > 4 ? atof(argv[4]) : 5.0;
  const double dt_ms = argc > 5 ? atof(argv[5]) : 0.5;
  const unsigned long long seed = argc > 6 ? strtoull(argv[6], nullptr, 10) : 1337ULL;
  const int stim_neurons = 500;

  std::mt19937_64 rng(seed);
  std::vector<int> stim_pop(stim_neurons);
  {
    std::uniform_int_distribution<int> ud(0, n - 1);
    for (int i = 0; i < stim_neurons; ++i) stim_pop[i] = ud(rng);
  }

  const double p_spike = stim_hz * dt_ms / 1000.0;
  std::vector<int> flat;
  std::vector<int> off(steps + 1, 0);
  std::uniform_real_distribution<double> ur(0.0, 1.0);
  for (int t = 0; t < steps; ++t) {
    for (int i = 0; i < stim_neurons; ++i)
      if (ur(rng) < p_spike) flat.push_back(stim_pop[i]);
    off[t + 1] = static_cast<int>(flat.size());
  }

  FILE* f = fopen(out, "wb");
  if (!f) { fprintf(stderr, "cannot write %s\n", out); return 2; }
  fwrite(&steps, sizeof(int), 1, f);
  fwrite(off.data(), sizeof(int), off.size(), f);
  if (!flat.empty()) fwrite(flat.data(), sizeof(int), flat.size(), f);
  fclose(f);
  printf("%s: n_neurons=%d steps=%d events=%zu seed=%llu\n", out, n, steps, flat.size(), seed);
  return 0;
}
