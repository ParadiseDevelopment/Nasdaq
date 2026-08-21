#!/usr/bin/env python3
"""
Validate a NAS100ML checkpoint before feeding it to MetaTrader.

The EA rejects a malformed checkpoint silently-ish (it logs and carries on
with freshly initialised weights), which is easy to miss in a live chart.
This script applies the same size rules as CScaler/CRffModel/CMlpModel
SetParams() and prints exactly what the EA would accept.

    python3 tools/verify_checkpoint.py model.txt
"""

import sys


def parse(path):
    with open(path, "r", encoding="ascii", errors="replace") as fh:
        lines = [ln.strip() for ln in fh]

    if not lines or "#NAS100ML" not in lines[0]:
        raise SystemExit(f"{path}: missing #NAS100ML header")

    nfeat = None
    sections = {}
    i = 1
    while i < len(lines):
        ln = lines[i]
        if not ln:
            i += 1
            continue
        if ln == "END":
            break
        parts = ln.split()
        if parts[0] == "VERSION":
            i += 1
            continue
        if parts[0] == "NFEAT":
            nfeat = int(parts[1])
            i += 1
            continue
        if parts[0] == "SECTION":
            tag, count = parts[1], int(parts[2])
            values = [float(v) for v in lines[i + 1: i + 1 + count]]
            if len(values) != count:
                raise SystemExit(f"{path}: section {tag} truncated "
                                 f"({len(values)} of {count} values)")
            sections[tag] = values
            i += 1 + count
            continue
        raise SystemExit(f"{path}: unexpected line {i + 1}: {ln!r}")

    if nfeat is None:
        raise SystemExit(f"{path}: no NFEAT line")
    return nfeat, sections


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)

    path = sys.argv[1]
    nfeat, sec = parse(path)
    print(f"{path}")
    print(f"  NFEAT = {nfeat}")

    problems = []

    def check(tag, expected, note=""):
        if tag not in sec:
            problems.append(f"{tag}: missing")
            print(f"  {tag:<9} missing")
            return
        got = len(sec[tag])
        ok = got == expected
        if not ok:
            problems.append(f"{tag}: {got} values, expected {expected}")
        print(f"  {tag:<9} {got:>7} values  expected {expected:>7}  "
              f"{'ok' if ok else 'MISMATCH'}{note}")

    check("SCALER", 2 * nfeat)
    check("LOGISTIC", nfeat + 1)

    if "RFF" in sec:
        D = int(round(sec["RFF"][0]))
        check("RFF", 2 + D * nfeat + 2 * D + 1, f"   (D={D})")
    else:
        check("RFF", -1)

    if "MLP" in sec:
        H = int(round(sec["MLP"][0]))
        check("MLP", 1 + H * nfeat + 2 * H + 1, f"   (H={H})")
    else:
        check("MLP", -1)

    check("HEDGE", 3)

    if "HEDGE" in sec:
        w = sec["HEDGE"]
        print(f"  hedge     lin={w[0]:.3f} rff={w[1]:.3f} mlp={w[2]:.3f} "
              f"(sum {sum(w):.3f})")

    if "SCALER" in sec:
        std = sec["SCALER"][nfeat:]
        bad = [i for i, s in enumerate(std) if s <= 0]
        if bad:
            problems.append(f"SCALER: non-positive std at features {bad}")

    print()
    if problems:
        print("REJECTED - the EA will not load this file cleanly:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)

    print("OK - every section matches what the EA expects.")


if __name__ == "__main__":
    main()
