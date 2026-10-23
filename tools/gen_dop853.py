import re

import numpy as np

with open('dop853_coefficients.py') as fh:
    src = fh.read()
ns = {}
exec(src, ns)
A_ref, B_ref, C_ref, E3_ref, E5_ref, D_ref = ns['A'], ns['B'], ns['C'], ns['E3'], ns['E5'], ns['D']
NE = 16

def rust_lit(s):
    # python/fortran decimal -> rust f64 literal (ensure a digit after '.', keep full precision)
    s = s.strip()
    return s

# --- parse A[i, j] = literal ---
Alit = [["0.0"]*NE for _ in range(NE)]
for m in re.finditer(r'A\[(\d+),\s*(\d+)\]\s*=\s*([-\d.eE+]+)', src):
    i, j, v = int(m[1]), int(m[2]), m[3]
    Alit[i][j] = v
# --- C from np.array([...]) ---
cm = re.search(r'C = np\.array\(\[(.*?)\]\)', src, re.S)
Clit = [x.strip() for x in cm[1].replace('\n',' ').split(',') if x.strip()]
# --- E5[j] = literal ---
E5lit = ["0.0"]*13
for m in re.finditer(r'E5\[(\d+)\]\s*=\s*([-\d.eE+]+)', src):
    E5lit[int(m[1])] = m[2]
# --- E3 adjustments: E3[0]-=, E3[8]-=, E3[11]-= ---
adj = {}
for m in re.finditer(r'E3\[(\d+)\]\s*-=\s*([-\d.eE+]+)', src):
    adj[int(m[1])] = m[2]
# E3[j] = A[12][j] (=B) minus adjustment if present
E3expr = []
for j in range(13):
    base = Alit[12][j] if j < 12 else "0.0"
    if j in adj:
        E3expr.append(f"{base} - {adj[j]}")
    else:
        E3expr.append(base)
# --- D[i, j] = literal ---
Dlit = [["0.0"]*NE for _ in range(4)]
for m in re.finditer(r'D\[(\d+),\s*(\d+)\]\s*=\s*([-\d.eE+]+)', src):
    Dlit[int(m[1])][int(m[2])] = m[3]

# B = A[12][0..11] literals
Blit = [Alit[12][j] for j in range(12)]

def emit_row(vals):
    return "[" + ", ".join(vals) + "]"

out = []
out.append("//! Machine-generated from scipy `dop853_coefficients.py` (BSD-3-Clause), whose values are the")
out.append("//! published Dormand-Prince 8(5,3) coefficients of Hairer-Norsett-Wanner (`dop853.f`). Emitted")
out.append("//! as the **full-precision decimal literals of the primary source** — Rust parses each to the")
out.append("//! correctly-rounded nearest f64, so no precision is lost and there is no double rounding. The")
out.append("//! ~1e-15 residual in the order conditions is the f64 floor, not truncation: the coefficients")
out.append("//! are irrational and given to ~30 digits. Regenerate with `tools/gen_dop853.py`; verified in")
out.append("//! `derivations/dop853_order.mac` and `tests/dop853.rs`. DO NOT hand-edit.")
out.append("")
out.append(f"pub(crate) const C: [f64; {len(Clit)}] = {emit_row(Clit)};")
out.append(f"pub(crate) const B: [f64; 12] = {emit_row(Blit)};")
out.append(f"pub(crate) const E5: [f64; 13] = {emit_row(E5lit)};")
out.append("/// `E3 = B` with three published adjustments (Hairer); the subtractions are exact f64 at compile time.")
out.append(f"pub(crate) const E3: [f64; 13] = {emit_row(E3expr)};")
out.append(f"pub(crate) const A: [[f64; {NE}]; {NE}] = [")
for i in range(NE):
    out.append("    " + emit_row(Alit[i]) + ",")
out.append("];")
out.append(f"pub(crate) const D: [[f64; {NE}]; 4] = [")
for i in range(4):
    out.append("    " + emit_row(Dlit[i]) + ",")
out.append("];")
with open('dop853_consts.rs', 'w') as fh:
    fh.write("\n".join(out) + "\n")

# --- self-check: parsed literals must reproduce scipy's f64 arrays bit-for-bit ---
def f(s):
    return float(s)
okA = all(np.float64(f(Alit[i][j])).view(np.int64)==np.float64(A_ref[i][j]).view(np.int64) for i in range(NE) for j in range(NE))
okC = all(np.float64(f(Clit[j])).view(np.int64)==np.float64(C_ref[j]).view(np.int64) for j in range(16))
okB = all(np.float64(f(Blit[j])).view(np.int64)==np.float64(B_ref[j]).view(np.int64) for j in range(12))
okE5= all(np.float64(f(E5lit[j])).view(np.int64)==np.float64(E5_ref[j]).view(np.int64) for j in range(13))
okD = all(np.float64(f(Dlit[i][j])).view(np.int64)==np.float64(D_ref[i][j]).view(np.int64) for i in range(4) for j in range(NE))
# E3: evaluate the expression in f64
def evalE3(e):
    if ' - ' in e:
        a, b = e.split(' - ')
        return f(a) - f(b)
    return f(e)
okE3= all(np.float64(evalE3(E3expr[j])).view(np.int64)==np.float64(E3_ref[j]).view(np.int64) for j in range(13))
print(f"bit-for-bit vs scipy:  A={okA} B={okB} C={okC} E3={okE3} E5={okE5} D={okD}")
print("all match" if all([okA,okB,okC,okE3,okE5,okD]) else "*** MISMATCH ***")
