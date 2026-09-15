// Small comparison driver for the sibling Legolas++ checkout.
//
// Compile from Interleave.jl with the same flags as Legolas++ itself:
//   c++ -O3 -DNDEBUG -std=c++14 -arch arm64 -fno-slp-vectorize \
//       -mtune=native -I../Interleave -I../Interleave/Interleave/include \
//       -I../Interleave/tst/MultiThomas \
//       bench/legolas_cpp_vectorization_audit.cxx -o /tmp/legolas_cpp_audit
// Add -ffp-contract=off when comparing against Interleave.jl's bit-exact path.

#define main legolas_multithomas_main
#include "../../Interleave/tst/MultiThomas/MultiThomas.cxx"
#undef main

template <int P>
struct AuditCase {
  using Array = typename std::conditional<
      P == 1, Interleave::Array<float, 2>,
      Interleave::Array<float, 2, P, 2>>::type;

  Array X, D, U, L, B;

  AuditCase(int nsys, int nx)
      : X(nsys, nx), D(nsys, nx), U(nsys, nx), L(nsys, nx), B(nsys, nx) {
    X.fill(0.0f);
    D.fill(2.0f);
    U.fill(-1.0f);
    L.fill(-1.0f);
    B.fill(1.0f);
  }

  void run() { Interleave::map(ThomasSolver(), D, U, L, B, X); }
};

template <int P>
double sample(AuditCase<P> &c, int inner) {
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < inner; ++i) c.run();
  const auto stop = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(stop - start).count() / inner;
}

int main(int argc, char **argv) {
  const int nx = argc > 1 ? std::atoi(argv[1]) : 64;
  const int nsys = nx * nx;
  const int rounds = 16;
  const int inner = 20;
  AuditCase<1> p1(nsys, nx);
  AuditCase<4> p4(nsys, nx);
  AuditCase<8> p8(nsys, nx);
  AuditCase<16> p16(nsys, nx);
  AuditCase<32> p32(nsys, nx);
  for (int i = 0; i < 3; ++i) {
    p1.run(); p4.run(); p8.run(); p16.run(); p32.run();
  }

  double best1 = 1e30, best4 = 1e30, best8 = 1e30, best16 = 1e30,
         best32 = 1e30;
  for (int r = 0; r < rounds; ++r) {
    // Rotate the first case so that slow load/frequency drift is not assigned
    // systematically to one packet size.
    switch (r % 5) {
      case 0:
        best1 = std::min(best1, sample(p1, inner));
        best4 = std::min(best4, sample(p4, inner));
        best8 = std::min(best8, sample(p8, inner));
        best16 = std::min(best16, sample(p16, inner));
        best32 = std::min(best32, sample(p32, inner));
        break;
      case 1:
        best4 = std::min(best4, sample(p4, inner));
        best8 = std::min(best8, sample(p8, inner));
        best16 = std::min(best16, sample(p16, inner));
        best32 = std::min(best32, sample(p32, inner));
        best1 = std::min(best1, sample(p1, inner));
        break;
      case 2:
        best8 = std::min(best8, sample(p8, inner));
        best16 = std::min(best16, sample(p16, inner));
        best32 = std::min(best32, sample(p32, inner));
        best1 = std::min(best1, sample(p1, inner));
        best4 = std::min(best4, sample(p4, inner));
        break;
      case 3:
        best16 = std::min(best16, sample(p16, inner));
        best32 = std::min(best32, sample(p32, inner));
        best1 = std::min(best1, sample(p1, inner));
        best4 = std::min(best4, sample(p4, inner));
        best8 = std::min(best8, sample(p8, inner));
        break;
      default:
        best32 = std::min(best32, sample(p32, inner));
        best1 = std::min(best1, sample(p1, inner));
        best4 = std::min(best4, sample(p4, inner));
        best8 = std::min(best8, sample(p8, inner));
        best16 = std::min(best16, sample(p16, inner));
    }
  }

  std::cout << "P=1  " << best1 * 1e3 << " ms  1x\n"
            << "P=4  " << best4 * 1e3 << " ms  " << best1 / best4 << "x\n"
            << "P=8  " << best8 * 1e3 << " ms  " << best1 / best8 << "x\n"
            << "P=16 " << best16 * 1e3 << " ms  " << best1 / best16 << "x\n"
            << "P=32 " << best32 * 1e3 << " ms  " << best1 / best32 << "x\n";
  return 0;
}
