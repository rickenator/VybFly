// Sparse event-driven LIF propagation over the FlyWire CSR (FlyScale phase-3 style
// workload, used as the instrumented payload for M4 energy measurement).
//
// Algorithm per biological timestep (device-side; the host synchronises twice per step to
// size the launches, which is the real cost structure of a bucketed DES step):
//   1. scatter: every neuron that fired on the previous step injects its synapses' weight
//      into the target neurons' accumulator I[] with atomicAdd; each touched target is
//      appended (dedup via a flag array) to the touched list; every *edge* walked counts as
//      one synaptic event.
//   2. stimulus scatter: the externally-driven (sensory) neurons scheduled for this step do
//      the same; each of them counts as one stimulus spike.
//   3. update: for every touched neuron, V = V*decay + I (leak applied only where input
//      arrived -- a documented event-driven approximation), threshold test, refractory gate,
//      fired neurons go into the next active list. Float atomicAdd ordering is not
//      deterministic in the last bits, so the harness *measures* run-to-run repeatability
//      (epoch-to-epoch counter deltas) instead of assuming it.
//
// The neuron model is NOT the project's biological baseline (agent C owns that); it exists
// to be a real, bounded, counter-rich sparse workload for power measurement.
//
// Build: nvcc -O3 -arch=sm_86 -o sparse_prop sparse_prop.cu
// Run:   ./sparse_prop --dir <canonical_dir> --epochs 8 --steps 1000 --dt-ms 0.5 ...
// Prints one "key=value" line prefixed RESULT, optionally writes --out <file>.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <vector>

#define CK(x)                                                                        \
  do {                                                                               \
    cudaError_t _e = (x);                                                            \
    if (_e != cudaSuccess) {                                                         \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__,  \
              __LINE__);                                                             \
      exit(2);                                                                       \
    }                                                                                \
  } while (0)

enum Counter { C_SPK = 0, C_STIM = 1, C_SYN = 2, C_OVER = 3, C_NCNT = 4 };
enum StepCnt { S_TOUCHED = 0, S_NEXT = 1, S_NCNT = 2 };

__global__ void scatter_kernel(const long long* __restrict__ ip,
                               const int* __restrict__ dst,
                               const float* __restrict__ w,
                               const int* __restrict__ act, int nact,
                               float* __restrict__ I, int* __restrict__ flag,
                               int* __restrict__ tlist,
                               unsigned long long* __restrict__ cum,
                               int* __restrict__ step_cnt,
                               int is_stim, int n_neurons) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= nact) return;
  int i = act[t];
  if (i < 0 || i >= n_neurons) return;
  if (is_stim) atomicAdd(&cum[C_STIM], 1ULL);
  long long s = ip[i], e = ip[i + 1];
  if (e <= s) return;
  atomicAdd(&cum[C_SYN], (unsigned long long)(e - s));
  for (long long k = s; k < e; ++k) {
    int j = dst[k];
    atomicAdd(&I[j], w[k]);
    if (atomicExch(&flag[j], 1) == 0) {
      int slot = atomicAdd(&step_cnt[S_TOUCHED], 1);
      if (slot < n_neurons) tlist[slot] = j;
    }
  }
}

__global__ void decay_kernel(float* __restrict__ V, int n_neurons, float decay) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_neurons) V[i] *= decay;
}

__global__ void update_kernel(const int* __restrict__ tlist, int nt,
                              float* __restrict__ V, int* __restrict__ refr,
                              float* __restrict__ I, int* __restrict__ flag,
                              int* __restrict__ next,
                              unsigned long long* __restrict__ cum,
                              int* __restrict__ step_cnt,
                              float vth, int step, int refr_steps,
                              int max_active) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= nt) return;
  int n = tlist[t];
  float v = V[n] + I[n];          // leak already applied to every neuron by decay_kernel
  I[n] = 0.0f;
  flag[n] = 0;
  if (v >= vth && step >= refr[n]) {
    V[n] = 0.0f;
    refr[n] = step + refr_steps;
    atomicAdd(&cum[C_SPK], 1ULL);
    int idx = atomicAdd(&step_cnt[S_NEXT], 1);
    if (idx < max_active) next[idx] = n;
    else atomicAdd(&cum[C_OVER], 1ULL);
  } else {
    V[n] = v;
  }
}

static std::string jpath(const std::string& dir, const char* name) {
  return dir + "/bin/" + name;
}

template <typename T>
static std::vector<T> read_bin(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(2); }
  std::streamsize bytes = f.tellg();
  f.seekg(0);
  std::vector<T> v((size_t)(bytes / (std::streamsize)sizeof(T)));
  f.read(reinterpret_cast<char*>(v.data()), bytes);
  if (!f) { fprintf(stderr, "short read %s\n", path.c_str()); exit(2); }
  return v;
}

int main(int argc, char** argv) {
  std::map<std::string, std::string> a;
  for (int i = 1; i + 1 < argc; i += 2) a[argv[i]] = argv[i + 1];
  auto geti = [&](const char* k, long long d) -> long long {
    auto it = a.find(k); return it == a.end() ? d : atoll(it->second.c_str()); };
  auto getf = [&](const char* k, double d) -> double {
    auto it = a.find(k); return it == a.end() ? d : atof(it->second.c_str()); };
  auto gets = [&](const char* k, const char* d) -> std::string {
    auto it = a.find(k); return it == a.end() ? std::string(d) : it->second; };

  const std::string dir = gets("--dir", ".");
  const int epochs = (int)geti("--epochs", 4);
  const int steps = (int)geti("--steps", 1000);
  const double dt_ms = getf("--dt-ms", 0.5);
  const unsigned long long seed = (unsigned long long)geti("--seed", 1337);
  const int stim_neurons = (int)geti("--stim-neurons", 500);
  const double stim_hz = getf("--stim-hz", 5.0);
  const float vth = (float)getf("--vth", 1.0);
  const float decay = (float)getf("--decay", 0.9);
  const double syn_gain = getf("--syn-gain", 1.0);
  const int refr_steps = (int)geti("--refr-steps", 5);
  const int max_active = (int)geti("--max-active", 300000);
  const int parity_steps = (int)geti("--parity-steps", 0);
  const std::string parity_out = gets("--parity-out", "");
  const std::string out_path = gets("--out", "");

  auto ip_h = read_bin<long long>(jpath(dir, "csr.out.indptr.i64"));
  auto dst_h = read_bin<int>(jpath(dir, "csr.out.indices.i32"));
  auto syn_h = read_bin<int>(jpath(dir, "csr.out.syn.i32"));
  const int n_neurons = (int)ip_h.size() - 1;
  const long long n_edges = (long long)dst_h.size();

  // Weights: either supplied by the harness (float32, n_edges, already carrying the
  // excitatory/inhibitory sign from the published transmitter annotation) or built here as
  // synapse-count-normalised excitatory weights scaled by --syn-gain. Every neuron's total
  // synaptic input sums to 1 in the built-in path, so a single firing presynaptic partner
  // can never drive a target over threshold on its own.
  std::vector<float> w_h;
  const std::string weights_file = gets("--weights", "");
  std::string weight_source;
  if (!weights_file.empty()) {
    w_h = read_bin<float>(weights_file);
    if ((long long)w_h.size() != n_edges) {
      fprintf(stderr, "weights file %s has %zu entries, expected %lld\n",
              weights_file.c_str(), w_h.size(), n_edges);
      exit(2);
    }
    weight_source = weights_file;
  } else {
    std::vector<double> total_in(n_neurons, 0.0);
    for (long long e = 0; e < n_edges; ++e) total_in[dst_h[e]] += (double)syn_h[e];
    w_h.resize(n_edges);
    for (long long e = 0; e < n_edges; ++e) {
      double t = total_in[dst_h[e]];
      w_h[e] = t > 0.0 ? (float)(syn_gain * (double)syn_h[e] / t) : 0.0f;
    }
    weight_source = "built_in_synapse_count_normalised";
  }

  // Deterministic stimulus schedule. Preferred source is a file written by the harness
  // (so the CPU/numpy reference and this CUDA path drive the network with *identical*
  // stimulus events and the parity check is a real numerical comparison):
  //   int32 n_steps, int32 offsets[n_steps+1], int32 neuron_ids[n_events]
  // Falls back to an internal mt19937_64-based schedule when no file is given.
  const std::string stim_file = gets("--stim-file", "");
  std::vector<int> stim_flat;
  std::vector<int> stim_off;
  int sched_steps = 0;
  std::string stimulus_source;
  if (!stim_file.empty()) {
    std::ifstream f(stim_file, std::ios::binary);
    if (!f) { fprintf(stderr, "cannot open stim file %s\n", stim_file.c_str()); exit(2); }
    int n_steps_f = 0;
    f.read(reinterpret_cast<char*>(&n_steps_f), sizeof(int));
    stim_off.resize(n_steps_f + 1);
    f.read(reinterpret_cast<char*>(stim_off.data()), sizeof(int) * (n_steps_f + 1));
    int n_ev = stim_off[n_steps_f];
    stim_flat.resize(n_ev);
    if (n_ev > 0) f.read(reinterpret_cast<char*>(stim_flat.data()), sizeof(int) * n_ev);
    if (!f) { fprintf(stderr, "short read %s\n", stim_file.c_str()); exit(2); }
    sched_steps = n_steps_f;
    stimulus_source = stim_file;
  } else {
    std::mt19937_64 rng(seed);
    std::vector<int> stim_pop(stim_neurons);
    { std::uniform_int_distribution<int> ud(0, n_neurons - 1);
      for (int i = 0; i < stim_neurons; ++i) stim_pop[i] = ud(rng); }
    const double p_spike = stim_hz * dt_ms / 1000.0;
    sched_steps = std::max(steps, parity_steps);  // schedule must cover both paths
    stim_off.assign(sched_steps + 1, 0);
    std::uniform_real_distribution<double> ur(0.0, 1.0);
    for (int t = 0; t < sched_steps; ++t) {
      for (int i = 0; i < stim_neurons; ++i)
        if (ur(rng) < p_spike) stim_flat.push_back(stim_pop[i]);
      stim_off[t + 1] = (int)stim_flat.size();
    }
    stimulus_source = "built_in_mt19937_64_seed_" + std::to_string(seed);
  }
  if (sched_steps < std::max(steps, parity_steps)) {
    fprintf(stderr, "stimulus schedule covers %d steps but %d are required\n", sched_steps,
            std::max(steps, parity_steps));
    exit(2);
  }

  // ---- device memory ----------------------------------------------------------------
  int *dst_d, *flag_d, *tlist_d, *nextA, *nextB, *refr_d, *step_cnt_d, *stim_d;
  long long* ip_d;
  float *w_d, *I_d, *V_d;
  unsigned long long* cum_d;
  CK(cudaMalloc(&ip_d, sizeof(long long) * ip_h.size()));
  CK(cudaMalloc(&dst_d, sizeof(int) * n_edges));
  CK(cudaMalloc(&w_d, sizeof(float) * n_edges));
  CK(cudaMalloc(&I_d, sizeof(float) * n_neurons));
  CK(cudaMalloc(&V_d, sizeof(float) * n_neurons));
  CK(cudaMalloc(&refr_d, sizeof(int) * n_neurons));
  CK(cudaMalloc(&flag_d, sizeof(int) * n_neurons));
  CK(cudaMalloc(&tlist_d, sizeof(int) * n_neurons));
  CK(cudaMalloc(&nextA, sizeof(int) * max_active));
  CK(cudaMalloc(&nextB, sizeof(int) * max_active));
  CK(cudaMalloc(&step_cnt_d, sizeof(int) * S_NCNT));
  CK(cudaMalloc(&cum_d, sizeof(unsigned long long) * C_NCNT));
  if (!stim_flat.empty()) CK(cudaMalloc(&stim_d, sizeof(int) * stim_flat.size()));
  else CK(cudaMalloc(&stim_d, sizeof(int)));
  CK(cudaMemcpy(ip_d, ip_h.data(), sizeof(long long) * ip_h.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dst_d, dst_h.data(), sizeof(int) * n_edges, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(w_d, w_h.data(), sizeof(float) * n_edges, cudaMemcpyHostToDevice));
  if (!stim_flat.empty())
    CK(cudaMemcpy(stim_d, stim_flat.data(), sizeof(int) * stim_flat.size(),
                  cudaMemcpyHostToDevice));

  const int TPB = 256;
  auto rungrid = [&](int n) { return (n + TPB - 1) / TPB; };
  int* cur = nextA;
  int* nxt = nextB;

  auto reset_state = [&]() {
    CK(cudaMemset(I_d, 0, sizeof(float) * n_neurons));
    CK(cudaMemset(V_d, 0, sizeof(float) * n_neurons));
    CK(cudaMemset(refr_d, 0, sizeof(int) * n_neurons));
    CK(cudaMemset(flag_d, 0, sizeof(int) * n_neurons));
  };

  // returns the number of neurons in the next active list after the last step
  auto run_epoch = [&](int n_steps) -> int {
    reset_state();
    int act_count = 0;
    cur = nextA; nxt = nextB;
    for (int step = 0; step < n_steps; ++step) {
      CK(cudaMemset(step_cnt_d, 0, sizeof(int) * S_NCNT));
      if (act_count > 0)
        scatter_kernel<<<rungrid(act_count), TPB>>>(ip_d, dst_d, w_d, cur, act_count, I_d,
                                                    flag_d, tlist_d, cum_d, step_cnt_d, 0,
                                                    n_neurons);
      int s0 = stim_off[step], s1 = stim_off[step + 1];
      if (s1 > s0)
        scatter_kernel<<<rungrid(s1 - s0), TPB>>>(ip_d, dst_d, w_d, stim_d + s0, s1 - s0,
                                                  I_d, flag_d, tlist_d, cum_d, step_cnt_d,
                                                  1, n_neurons);
      int sc[S_NCNT] = {0, 0};
      CK(cudaMemcpy(sc, step_cnt_d, sizeof(int) * S_NCNT, cudaMemcpyDeviceToHost));
      decay_kernel<<<rungrid(n_neurons), TPB>>>(V_d, n_neurons, decay);
      if (sc[S_TOUCHED] > 0)
        update_kernel<<<rungrid(sc[S_TOUCHED]), TPB>>>(tlist_d, sc[S_TOUCHED], V_d, refr_d,
                                                       I_d, flag_d, nxt, cum_d, step_cnt_d,
                                                       vth, step, refr_steps, max_active);
      int nxt_count = 0;
      CK(cudaMemcpy(&nxt_count, step_cnt_d + S_NEXT, sizeof(int), cudaMemcpyDeviceToHost));
      if (nxt_count > max_active) nxt_count = max_active;
      int* tmp = cur; cur = nxt; nxt = tmp;   // next step reads what update just wrote
      act_count = nxt_count;
    }
    return act_count;
  };

  auto read_cum = [&](unsigned long long* out) {
    CK(cudaMemcpy(out, cum_d, sizeof(unsigned long long) * C_NCNT, cudaMemcpyDeviceToHost));
  };
  auto cleanup = [&]() {
    CK(cudaFree(ip_d)); CK(cudaFree(dst_d)); CK(cudaFree(w_d)); CK(cudaFree(I_d));
    CK(cudaFree(V_d)); CK(cudaFree(refr_d)); CK(cudaFree(flag_d)); CK(cudaFree(tlist_d));
    CK(cudaFree(nextA)); CK(cudaFree(nextB)); CK(cudaFree(step_cnt_d)); CK(cudaFree(cum_d));
    CK(cudaFree(stim_d));
  };

  // ---- parity path: one short deterministic epoch, counters only --------------------
  if (parity_steps > 0) {
    CK(cudaMemset(cum_d, 0, sizeof(unsigned long long) * C_NCNT));
    run_epoch(parity_steps);
    unsigned long long c[C_NCNT];
    read_cum(c);
    const char* fmt = "spikes=%llu stimulus_spikes=%llu synaptic_events=%llu overflow=%llu "
                      "steps=%d vth=%.6f decay=%.6f syn_gain=%.6f refr_steps=%d n_neurons=%d "
                      "n_edges=%lld n_active_end=%d\n";
    if (!parity_out.empty()) {
      FILE* fh = fopen(parity_out.c_str(), "w");
      if (!fh) { fprintf(stderr, "cannot write %s\n", parity_out.c_str()); exit(2); }
      fprintf(fh, fmt, c[C_SPK], c[C_STIM], c[C_SYN], c[C_OVER], parity_steps, (double)vth,
              (double)decay, syn_gain, refr_steps, n_neurons, n_edges, 0);
      fclose(fh);
    } else {
      printf("RESULT ");
      printf(fmt, c[C_SPK], c[C_STIM], c[C_SYN], c[C_OVER], parity_steps, (double)vth,
             (double)decay, syn_gain, refr_steps, n_neurons, n_edges, 0);
    }
    printf("STIMULUS source=%s events=%zu steps=%d\n", stimulus_source.c_str(),
           stim_flat.size(), sched_steps);
    cleanup();
    return 0;
  }

  // ---- timed multi-epoch run --------------------------------------------------------
  cudaEvent_t t0e, t1e;
  CK(cudaEventCreate(&t0e)); CK(cudaEventCreate(&t1e));
  CK(cudaMemset(cum_d, 0, sizeof(unsigned long long) * C_NCNT));
  std::vector<std::vector<unsigned long long>> per_epoch;
  CK(cudaEventRecord(t0e));
  for (int ep = 0; ep < epochs; ++ep) {
    run_epoch(steps);
    unsigned long long c[C_NCNT];
    read_cum(c);
    per_epoch.push_back({c[C_SPK], c[C_STIM], c[C_SYN], c[C_OVER]});
  }
  CK(cudaEventRecord(t1e));
  CK(cudaEventSynchronize(t1e));
  float kernel_ms = 0.0f;
  CK(cudaEventElapsedTime(&kernel_ms, t0e, t1e));

  const unsigned long long* tot = per_epoch.back().data();
  std::vector<std::vector<long long>> deltas;
  std::vector<unsigned long long> prev(C_NCNT, 0);
  bool identical = true;
  for (auto& row : per_epoch) {
    std::vector<long long> d(C_NCNT);
    for (int k = 0; k < C_NCNT; ++k) { d[k] = (long long)(row[k] - prev[k]); prev[k] = row[k]; }
    if (!deltas.empty() && d != deltas[0]) identical = false;
    deltas.push_back(d);
  }
  const double bio_seconds = (double)epochs * steps * dt_ms / 1000.0;

  printf("RESULT epochs=%d steps_per_epoch=%d steps_total=%d dt_ms=%.6f bio_seconds=%.6f "
         "n_neurons=%d n_edges=%lld vth=%.6f decay=%.6f syn_gain=%.6f refr_steps=%d "
         "stim_neurons=%d stim_hz=%.4f seed=%llu max_active=%d kernel_s=%.6f "
         "spikes=%llu stimulus_spikes=%llu synaptic_events=%llu overflow=%llu "
         "epochs_identical=%d\n",
         epochs, steps, epochs * steps, dt_ms, bio_seconds, n_neurons, n_edges, (double)vth,
         (double)decay, syn_gain, refr_steps, stim_neurons, stim_hz, seed, max_active,
         kernel_ms / 1000.0, tot[C_SPK], tot[C_STIM], tot[C_SYN], tot[C_OVER],
         identical ? 1 : 0);

  printf("WEIGHTS source=%s n_edges=%lld inhibitory_edges=-1(none)\n", weight_source.c_str(), n_edges);
  printf("STIMULUS source=%s events=%zu steps=%d\n", stimulus_source.c_str(), stim_flat.size(), sched_steps);

  if (!out_path.empty()) {
    FILE* fh = fopen(out_path.c_str(), "w");
    if (!fh) { fprintf(stderr, "cannot write %s\n", out_path.c_str()); exit(2); }
    fprintf(fh, "epochs=%d steps=%d bio_seconds=%.6f spikes=%llu stimulus_spikes=%llu "
                "synaptic_events=%llu overflow=%llu kernel_s=%.6f epochs_identical=%d\n",
            epochs, steps, bio_seconds, tot[C_SPK], tot[C_STIM], tot[C_SYN], tot[C_OVER],
            kernel_ms / 1000.0, identical ? 1 : 0);
    for (size_t i = 0; i < deltas.size(); ++i)
      fprintf(fh, "epoch_delta %zu spikes=%lld stimulus_spikes=%lld synaptic_events=%lld "
                  "overflow=%lld\n", i, deltas[i][C_SPK], deltas[i][C_STIM],
              deltas[i][C_SYN], deltas[i][C_OVER]);
    fclose(fh);
  }

  cleanup();
  return 0;
}
