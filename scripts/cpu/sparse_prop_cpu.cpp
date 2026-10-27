// sparse_prop_cpu.cpp - deterministic CPU reference for scripts/cuda/sparse_prop.cu.
//
// Same model, same parameters, same weights (built-in synapse-count-normalised, computed in
// double exactly as the CUDA host code does) and the same mt19937_64 stimulus schedule. The
// one thing the GPU leaves unspecified -- the order of float accumulation, which atomicAdd
// makes nondeterministic -- is fixed here to a canonical order:
//   * the active list of a step is the previous step's fired neurons in ASCENDING id order
//     (capped at --max-active, the first max_active ids kept);
//   * a step's input is accumulated source by source over (active ++ stimulus-of-this-step),
//     each source's out-edges in CSR order.
// That order is what the multithreaded Vyb engine (src/vyb/ladder_prop.vyb) reproduces bit for
// bit at any thread count, so this program is its exact oracle; vs the GPU it is a statistical
// match (spike totals within the 1% lif_baseline tolerance).
//
// Build: g++ -O3 -march=native -ffp-contract=off -o sparse_prop_cpu sparse_prop_cpu.cpp
// Run:   ./sparse_prop_cpu --dir <csr_dir> --epochs 8 --steps 1000 --syn-gain 6 --refr-steps 20
//        [--dump-stim <file>]   write the stimulus schedule in sparse_prop.cu's --stim-file format
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <vector>

template <typename T>
static std::vector<T> read_bin(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(2); }
  std::streamsize bytes = f.tellg();
  f.seekg(0);
  std::vector<T> v((size_t)(bytes / (std::streamsize)sizeof(T)));
  f.read(reinterpret_cast<char*>(v.data()), bytes);
  return v;
}

// Order-insensitive spike-set digest (shared definition with ladder_prop.vyb; no overflow).
static inline void digest(int64_t key, int64_t& sum, int64_t& x) {
  int64_t lo = key & 0x7FFFFFFF;
  sum += lo;
  x ^= (((key ^ (key >> 17)) & 0x7FFFFFFF) * 1000003 + (key >> 31)) & 0x3FFFFFFFFFFFFFFF;
}

int main(int argc, char** argv) {
  std::map<std::string, std::string> a;
  for (int i = 1; i + 1 < argc; i += 2) a[argv[i]] = argv[i + 1];
  auto geti = [&](const char* k, long long d) { auto it = a.find(k); return it == a.end() ? d : atoll(it->second.c_str()); };
  auto getf = [&](const char* k, double d) { auto it = a.find(k); return it == a.end() ? d : atof(it->second.c_str()); };
  auto gets = [&](const char* k, const char* d) { auto it = a.find(k); return it == a.end() ? std::string(d) : it->second; };
  const std::string dir = gets("--dir", ".");
  const int epochs = (int)geti("--epochs", 4), steps = (int)geti("--steps", 1000);
  const double dt_ms = getf("--dt-ms", 0.5);
  const unsigned long long seed = (unsigned long long)geti("--seed", 1337);
  const int stim_neurons = (int)geti("--stim-neurons", 500);
  const double stim_hz = getf("--stim-hz", 5.0);
  const float vth = (float)getf("--vth", 1.0), decay = (float)getf("--decay", 0.9);
  const double syn_gain = getf("--syn-gain", 1.0);
  const int refr_steps = (int)geti("--refr-steps", 5), max_active = (int)geti("--max-active", 300000);
  const std::string dump_stim = gets("--dump-stim", "");

  auto t_load0 = std::chrono::steady_clock::now();
  auto ip = read_bin<long long>(dir + "/bin/csr.out.indptr.i64");
  auto dst = read_bin<int>(dir + "/bin/csr.out.indices.i32");
  auto syn = read_bin<int>(dir + "/bin/csr.out.syn.i32");
  const int n = (int)ip.size() - 1;
  const long long ne = (long long)dst.size();
  std::vector<double> total_in(n, 0.0);
  for (long long e = 0; e < ne; ++e) total_in[dst[e]] += (double)syn[e];
  std::vector<float> w(ne);
  for (long long e = 0; e < ne; ++e) {
    double t = total_in[dst[e]];
    w[e] = t > 0.0 ? (float)(syn_gain * (double)syn[e] / t) : 0.0f;
  }
  // stimulus: byte-for-byte the CUDA fallback schedule
  std::vector<int> stim_flat, stim_off;
  {
    std::mt19937_64 rng(seed);
    std::vector<int> pop(stim_neurons);
    { std::uniform_int_distribution<int> ud(0, n - 1); for (int i = 0; i < stim_neurons; ++i) pop[i] = ud(rng); }
    const double p = stim_hz * dt_ms / 1000.0;
    stim_off.assign(steps + 1, 0);
    std::uniform_real_distribution<double> ur(0.0, 1.0);
    for (int t = 0; t < steps; ++t) {
      for (int i = 0; i < stim_neurons; ++i) if (ur(rng) < p) stim_flat.push_back(pop[i]);
      stim_off[t + 1] = (int)stim_flat.size();
    }
  }
  if (!dump_stim.empty()) {
    FILE* fh = fopen(dump_stim.c_str(), "wb");
    int ns = steps; fwrite(&ns, 4, 1, fh);
    fwrite(stim_off.data(), 4, stim_off.size(), fh);
    if (!stim_flat.empty()) fwrite(stim_flat.data(), 4, stim_flat.size(), fh);
    fclose(fh);
  }
  double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load0).count();

  std::vector<float> I(n), V(n);
  std::vector<int> refr(n), act, nxt;
  std::vector<unsigned char> flag(n);
  act.reserve(n); nxt.reserve(n);
  unsigned long long spk = 0, stim = 0, synev = 0, over = 0;
  int64_t hsum = 0, hx = 0;
  std::vector<unsigned long long> ep_spk;
  auto t0 = std::chrono::steady_clock::now();
  for (int ep = 0; ep < epochs; ++ep) {
    std::fill(I.begin(), I.end(), 0.0f); std::fill(V.begin(), V.end(), 0.0f);
    std::fill(refr.begin(), refr.end(), 0); std::fill(flag.begin(), flag.end(), 0);
    act.clear();
    unsigned long long spk0 = spk;
    for (int step = 0; step < steps; ++step) {
      auto push_src = [&](int i) {
        long long s = ip[i], e = ip[i + 1];
        if (e <= s) return;
        synev += (unsigned long long)(e - s);
        for (long long k = s; k < e; ++k) { int j = dst[k]; I[j] += w[k]; flag[j] = 1; }
      };
      for (int i : act) push_src(i);
      for (int q = stim_off[step]; q < stim_off[step + 1]; ++q) {
        int i = stim_flat[q];
        if (i < 0 || i >= n) continue;
        ++stim; push_src(i);
      }
      nxt.clear();
      for (int i = 0; i < n; ++i) {
        float vd = V[i] * decay;
        if (flag[i]) {
          float v = vd + I[i];
          I[i] = 0.0f; flag[i] = 0;
          if (v >= vth && step >= refr[i]) {
            V[i] = 0.0f; refr[i] = step + refr_steps; ++spk;
            if (ep == 0) digest((int64_t)step * n + i, hsum, hx);
            if ((int)nxt.size() < max_active) nxt.push_back(i); else ++over;
          } else V[i] = v;
        } else V[i] = vd;
      }
      act.swap(nxt);
    }
    ep_spk.push_back(spk - spk0);
  }
  double sim_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  bool ident = true;
  for (auto v : ep_spk) if (v != ep_spk[0]) ident = false;
  printf("RESULT engine=cpu_ref threads=1 epochs=%d steps=%d n_neurons=%d n_edges=%lld load_s=%.6f "
         "sim_s=%.6f spikes=%llu stimulus_spikes=%llu synaptic_events=%llu overflow=%llu "
         "epochs_identical=%d epoch0_spikes=%llu digest_sum=%lld digest_xor=%lld\n",
         epochs, steps, n, ne, load_s, sim_s, spk, stim, synev, over, ident ? 1 : 0, ep_spk[0],
         (long long)hsum, (long long)hx);
  return 0;
}
