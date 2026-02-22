# phase3p1 + GPVM tests & step-(2) runner

## 1) Run tests (unit + regression)

From this directory:

```bash
./run_tests.sh
```

Behavior:
- If `pytest` is available, runs `python -m pytest -q tests_gpvm`.
- If `pytest` is NOT available (offline / minimal env), runs:

```bash
python run_tests_no_pytest.py
```

## 2) Step-(2): phase3 optimization -> GPVM EML profile

Run:

```bash
python run_phase3_opt_then_gpvm_eml_profile.py
```

Outputs are written to:

```text
./out_gpvm_step2/
  best_geometry.json
  best_geometry.json

  # (A) phase3-compatible BT-cavity profile sampled over physical EML
  z_eml_phys_nm.npy
  I_iso_phase3BT.npy
  I_TE_h_phase3BT.npy
  I_TM_h_phase3BT.npy
  I_TM_v_phase3BT.npy
  gpvm_eml_profiles_phase3BT.png

  # (B) strict-EML diagnostic profile (physical EML boundaries)
  z_eml_nm.npy
  I_TE_h_strictEML.npy
  I_TM_h_strictEML.npy
  I_TM_v_strictEML.npy
  I_iso_strictEML.npy
  gpvm_eml_profiles_strictEML.png
```

Notes:
- In this repository, TE uses tangential **E** amplitude, TM uses tangential
  **H** amplitude. This matches the phase3 convention (and explains why TM
  reflection amplitudes differ by a sign from many "E-amplitude" optics codes).
- If you want the resonance denominator to match the phase3 F(lambda) proxy,
  use the (A) phase3-compatible output (`*_phase3BT.*`). The strict-EML profile
  is kept as a diagnostic, because it uses a different effective cavity
  definition (EML boundaries instead of BT terminals).

## 3) Bottom PEC-proxy diagnostics (k sweep + LCAV profile)

This reproduces the discussion plots about making the **bottom** boundary
"PEC-like" by increasing the imaginary refractive index `k` of a semi-infinite
metal ambient (numerically stable proxy for PEC).

Run:

```bash
python run_bottom_metal_PEC_proxy.py
```

Outputs:

```text
./out_bottom_metal/
  need_k_for_E0_small.png
  need_k_for_E0_small.csv
  gpvm_lcav_profile_bottomMetal_k2000.png
```
