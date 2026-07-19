import subprocess
import csv

def ncu_func(B, H, S, D, causal, precision, is_orig):
    bin_ = f"./fp{precision}-fwd-bench-orig " if is_orig else f"./fp{precision}-fwd-bench "
    cmd = (
        "ncu --set full "
        "--launch-skip 100 "
        "--launch-count 3 "
        f"{bin_}"
        "--single "
        f"--batch-size {str(B)} "
        f"--num-heads {str(H)} "
        f"--seq-len {str(S)} "
        f"--head-dim {str(D)} "
        f"--causal {str(causal)} "
        "--warmup 100 "
        "--iter 10"
    )
    print(cmd)
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True, 
        text=True
    )
    log =  result.stdout
    if "ERROR" in log:
        return None
    return log

def read_csv(file_dir="./fc.csv"):
    f = open(file_dir, "r")
    reader_ = csv.reader(f)
    rows = []
    for row in reader_:
        if row[0] != "B":
            p = "8" if row[-1] == "fp8" else "16"
            r = row.copy()
            r[-1] = p
            rows.append(r)
    f.close()
    return rows

def main():
    for row in read_csv():
        B, H, S, D, causal, precision = row
        for t in [False, True]:
            result = ncu_func(B, H, S, D, causal, precision, t)
            if result is not None:
                sn = f"{B}-{H}-{S}-{D}-causal_{causal}-fp{precision}-fa3.txt" if t else f"{B}-{H}-{S}-{D}-causal_{causal}-fp{precision}-fat.txt"
                print(f"[SAVE]: {sn}")
                f = open(f"./ncu_files/{sn}", "w")
                f.write(result)
                f.close()


if __name__ == "__main__":
    main()