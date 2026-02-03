import math
import argparse
import numpy as np

def _autocorr_fft(x):
    """
    Normalized autocorrelation function rho(t) for a 1D series x,
    computed with FFT. Returns rho with rho[0] = 1.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 2:
        return np.ones(1)

    x_centered = x - x.mean()
    # zero-pad to 2*n to avoid wraparound in first n lags
    f = np.fft.rfft(x_centered, n=2 * n)
    acf = np.fft.irfft(f * np.conjugate(f), n=2 * n)[:n].real
    acf /= acf[0]  # normalize: rho(0) = 1
    return acf


def sem_from_sokal(x, c=5.0):
    """
    Sokal-style windowed integrated autocorrelation time τ_int:

      τ_int(W) = 1/2 + sum_{t=1..W} rho(t)

    Choose W such that W > c * τ_int(W) (c ~ 4–10; default 5).
    Then:

      g_raw = 2 τ_int              (statistical inefficiency)
      g     = max(g_raw, 1.0)      (enforce g >= 1 so SEM_Sokal >= SEM_naive)
      n_eff = N / g
      SEM   = s / sqrt(n_eff)

    Returns (sem, n_eff, tau_int_est).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 2:
        return math.nan, math.nan, math.nan

    rho = _autocorr_fft(x)

    tau_int = 0.5
    W_star = 1
    for W in range(1, n):
        tau_int = 0.5 + np.sum(rho[1:W + 1])
        if W > c * tau_int:
            W_star = W
            break
        W_star = W  # fallback if we never break

    g_raw = 2.0 * tau_int
    g = max(g_raw, 1.0)
    n_eff = n / g

    std_val = x.std(ddof=1)
    sem_val = std_val / math.sqrt(n_eff)
    return sem_val, n_eff, tau_int


def read_n_columns(path, ncols=6):
    cols = [[] for _ in range(ncols)]
    with open(path, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < ncols:
                continue
            try:
                for j in range(ncols):
                    cols[j].append(float(parts[j]))
            except ValueError:
                continue
    return [np.asarray(c, dtype=float) for c in cols]


def main():
    ap = argparse.ArgumentParser(description="Compute Sokal SEM for 6-column helanal log.")
    ap.add_argument("log_file", help="Path to helanal_C2.log (6 whitespace-separated columns).")
    ap.add_argument("--c", type=float, default=5.0, help="Sokal window constant (default 5.0).")
    ap.add_argument("--ncols", type=int, default=6, help="Number of columns (default 6).")
    args = ap.parse_args()

    cols_arr = read_n_columns(args.log_file, ncols=args.ncols)
    n_rows = len(cols_arr[0]) if cols_arr else 0
    print(f"Input file: {args.log_file}")
    print(f"Number of lines with {args.ncols} columns: {n_rows}")

    per_col_stats = []
    for j, arr in enumerate(cols_arr):
        mean_j = float(arr.mean())
        std_j = float(arr.std(ddof=1))
        n_j = arr.size
        naive_sem_j = std_j / math.sqrt(n_j)

        sokal_sem_j, n_eff_j, tau_int_j = sem_from_sokal(arr, c=args.c)

        per_col_stats.append(
            {
                "mean": mean_j,
                "std": std_j,
                "sem_naive": naive_sem_j,
                "sem_sokal": sokal_sem_j,
                "n_eff": n_eff_j,
                "tau_int": tau_int_j,
            }
        )

        print(f"\nColumn {j+1}:")
        print(f"  Mean                 = {mean_j:.6f}")
        print(f"  Std (sample)         = {std_j:.6f}")
        print(f"  Naive SEM            = {naive_sem_j:.6f}")
        print(f"  Sokal SEM (>= naive) = {sokal_sem_j:.6f}")
        print(f"  n_eff (Sokal)        = {n_eff_j:.1f}")
        print(f"  tau_int (Sokal)      = {tau_int_j:.2f}")

    # Overall mean & overall SEM
    all_values = np.concatenate(cols_arr)
    overall_mean = float(all_values.mean())
    overall_std = float(all_values.std(ddof=1))
    N_total = all_values.size
    overall_naive_sem = overall_std / math.sqrt(N_total)

    sokal_sems = np.array([st["sem_sokal"] for st in per_col_stats], dtype=float)
    overall_sokal_sem = float(np.nanmax(sokal_sems))

    print("\n===== OVERALL =====")
    print(f"Overall mean (all columns, all rows):   {overall_mean:.6f}")
    print(f"Overall naive SEM (flattened):          {overall_naive_sem:.6f}")
    print(f"Overall Sokal SEM (max over columns):   {overall_sokal_sem:.6f}")


if __name__ == "__main__":
    main()
