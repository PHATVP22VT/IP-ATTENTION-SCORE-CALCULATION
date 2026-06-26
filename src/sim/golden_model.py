import numpy as np
import os

# ==========================================
# 1. CẤU HÌNH THÔNG SỐ (Hệ thống Q8.8)
#    TOKENS, D_HEAD, RUN_MODE giờ có thể nhập từ bàn phím
#    khi chạy (xem mục 7B). Giá trị dưới đây chỉ là DEFAULT
#    — dùng khi người dùng nhấn Enter bỏ qua (không nhập gì).
# ==========================================
TOKENS     = 64
D_HEAD     = 64    # chiều embedding mỗi head (= COLS của matmul)
DATA_WIDTH = 16
FRAC_BITS  = 8     # 1.0 thực = 256 đơn vị nguyên
ROM_DEPTH  = 2048

# ==========================================
# 2. MODE CHỌN DATA (default — sẽ được hỏi lại từ bàn phím)
#    RUN_MODE = 1 → Uniform: tất cả phần tử = UNIFORM_VAL
#    RUN_MODE = 2 → Random : seed=42, range [-0.3, 0.3]
# ==========================================
RUN_MODE    = 2
UNIFORM_VAL = 50   # chỉ dùng khi RUN_MODE = 1

# ==========================================
# 3. ĐƯỜNG DẪN XUẤT FILE (1 thư mục duy nhất)
# ==========================================
COE_OUT_PATH = r"E:\DOWNLOAD\HCMUT\TTKS\src\coe files\golden model"
MEM_OUT_PATH = r"E:\DOWNLOAD\HCMUT\TTKS\src\mem files\golden model"

# ==========================================
# 4. HÀM TIỆN ÍCH
# ==========================================
def get_int_input(prompt, default, min_val=None, max_val=None):
    """
    Nhập 1 số nguyên từ bàn phím.
    - Enter (bỏ trống) -> dùng giá trị default.
    - Nhập sai định dạng hoặc ngoài [min_val, max_val] -> hỏi lại.
    """
    while True:
        raw = input(f"{prompt} [mặc định = {default}]: ").strip()
        if raw == "":
            return default
        try:
            val = int(raw)
        except ValueError:
            print("  -> Vui lòng nhập một số nguyên hợp lệ.")
            continue
        if min_val is not None and val < min_val:
            print(f"  -> Giá trị phải >= {min_val}.")
            continue
        if max_val is not None and val > max_val:
            print(f"  -> Giá trị phải <= {max_val}.")
            continue
        return val


def float_to_q88(val_array):
    """Chuyển đổi mảng số thực sang số nguyên Q8.8 (16-bit có dấu)"""
    q88_vals = np.round(val_array * (1 << FRAC_BITS))
    return np.clip(q88_vals, -32768, 32767).astype(np.int64)

def write_coe_16(filename, data):
    """Ghi dữ liệu 16-bit ra .coe — dùng cho exp_rom"""
    try:
        os.makedirs(COE_OUT_PATH, exist_ok=True)
        filepath = os.path.join(COE_OUT_PATH, filename)
        with open(filepath, 'w') as f:
            f.write("memory_initialization_radix=16;\n")
            f.write("memory_initialization_vector=\n")
            for i, val in enumerate(data):
                sep = ";" if i == len(data) - 1 else ","
                f.write(f"{int(val) & 0xFFFF:04X}{sep}\n")
        print(f"[OK] COE 16-bit: {filepath}")
    except Exception as e:
        print(f"[ERROR] {filename}: {e}")

def write_coe_32(filename, data):
    """Ghi dữ liệu 32-bit ra .coe — dùng cho 3 RAM (q, k, v)"""
    try:
        os.makedirs(COE_OUT_PATH, exist_ok=True)
        filepath = os.path.join(COE_OUT_PATH, filename)
        with open(filepath, 'w') as f:
            f.write("memory_initialization_radix=16;\n")
            f.write("memory_initialization_vector=\n")
            for i, val in enumerate(data):
                sep = ";" if i == len(data) - 1 else ","
                f.write(f"{int(val) & 0xFFFFFFFF:08X}{sep}\n")
        print(f"[OK] COE 32-bit: {filepath}")
    except Exception as e:
        print(f"[ERROR] {filename}: {e}")

def write_mem_16(filename, data):
    """Ghi dữ liệu 16-bit Hex (.mem) — dùng cho $readmemh"""
    try:
        os.makedirs(MEM_OUT_PATH, exist_ok=True)
        filepath = os.path.join(MEM_OUT_PATH, filename)
        with open(filepath, 'w') as f:
            for val in data:
                f.write(f"{int(val) & 0xFFFF:04X}\n")
        print(f"[OK] MEM 16-bit: {filepath}")
    except Exception as e:
        print(f"[ERROR] {filename}: {e}")

def write_mem_32(filename, data):
    """Ghi dữ liệu 32-bit Hex (.mem) — dùng cho $readmemh và S01_AXI load"""
    try:
        os.makedirs(MEM_OUT_PATH, exist_ok=True)
        filepath = os.path.join(MEM_OUT_PATH, filename)
        with open(filepath, 'w') as f:
            for val in data:
                f.write(f"{int(val) & 0xFFFFFFFF:08X}\n")
        print(f"[OK] MEM 32-bit: {filepath}")
    except Exception as e:
        print(f"[ERROR] {filename}: {e}")

def write_golden_score(Score_int):
    """Xuất golden_score.mem (32-bit, TOKENS×TOKENS phần tử)"""
    try:
        os.makedirs(MEM_OUT_PATH, exist_ok=True)
        path = os.path.join(MEM_OUT_PATH, "golden_score.mem")
        with open(path, 'w') as f:
            for v in Score_int.flatten():
                f.write(f"{int(v) & 0xFFFFFFFF:08X}\n")
        print(f"[OK] golden_score.mem: {path}")
    except Exception as e:
        print(f"[ERROR] golden_score.mem: {e}")

# ==========================================
# 5. TẠO BẢNG TRA CỨU EXP LUT
# ==========================================
def generate_exp_lut():
    lut = []
    for i in range(ROM_DEPTH):
        x     = -i / (1 << FRAC_BITS)
        val   = np.exp(x)
        q_val = int(np.round(val * (1 << FRAC_BITS)))
        if q_val == 0 and val > 0:
            q_val = 1
        lut.append(q_val)
    return lut

# ==========================================
# 6. HÀM TÍNH TOÁN GOLDEN (BIT-TRUE)
#
#  Pipeline mới (bỏ phase Linear):
#    Đầu vào : Q_int [TOKENS × D_HEAD]  — đã ở dạng Q8.8
#              K_int [TOKENS × D_HEAD]  — đã ở dạng Q8.8
#              V_int [TOKENS × D_HEAD]  — đã ở dạng Q8.8 (dùng ở bước sau)
#
#    Phase 1 : Attention Score = Q × K^T  (phần tử = dot-product Q8.8 × Q8.8)
#              pe_unit: MAC rồi shift FRAC_BITS=8 → ra Q8.8
#              qk_controller: shift thêm SQRT_SHIFT=3 (≈ /√64 = /8)
#              Score_int = round(mac_sum / 2048)
#                        = round(Q_int · K_int^T / 256 / 8)
#              Tương đương HW: pe_unit shift 8 rồi qk_controller shift 3
#
#    Phase 2 : Softmax (giữ nguyên)
# ==========================================
def run_golden(Q_int, K_int, V_int, exp_lut_data, label=""):
    """
    Chạy toàn bộ pipeline golden model.
    Q_int, K_int, V_int: integer Q8.8 arrays shape [TOKENS × D_HEAD]
    """
    # ── Phase 1: Attention Score  Q × K^T / √d_head ──────────────────
    # mac_sum[i][j] = sum_k Q_int[i][k] * K_int[j][k]  (raw integer product)
    # pe_unit làm tròn sau FRAC_BITS=8 bit
    # qk_controller shift SQRT_SHIFT = log2(D_HEAD)//2 = 3 bit
    # → Score_int = round(mac_sum / 256) >> 3 = round(mac_sum / 2048)
    SQRT_SHIFT = int(np.log2(D_HEAD)) // 2          # = 3 khi D_HEAD=64
    mac_sum    = np.dot(Q_int, K_int.T)              # shape [TOKENS × TOKENS]
    pe_out    = np.round(mac_sum / 256.0).astype(np.int64)
    Score_int = (pe_out >> SQRT_SHIFT)  
    # Tương đương: Score_int = round(mac_sum / 2048)

    # ── Phase 2: Softmax — Max, Z, Exp LUT ───────────────────────────
    exp_lut_array = np.array(exp_lut_data)
    max_score     = np.max(Score_int, axis=1, keepdims=True)
    Z_int         = Score_int - max_score
    exp_Z_int     = np.zeros_like(Z_int)
    for i in range(Z_int.shape[0]):
        for j in range(Z_int.shape[1]):
            val = Z_int[i, j]
            if val <= 0:
                addr = (-val) & 0x7FF
                exp_Z_int[i, j] = exp_lut_array[addr] if addr < ROM_DEPTH else 0

    sum_exp_int    = np.sum(exp_Z_int, axis=1, keepdims=True)
    weights_int_2d = (exp_Z_int * 256) // sum_exp_int

    # ── In kết quả ───────────────────────────────────────────────────
    tag         = f"[{label}] " if label else ""
    Score_flat  = Score_int.flatten()
    exp_Z_flat  = exp_Z_int.flatten()
    weights_int = weights_int_2d.flatten()

    print("\n" + "="*65)
    print(f"  {tag}BANG DOI CHIEU GOLDEN MODEL (PYTHON) VS HARDWARE (VERILOG)")
    print("="*65)

    print(f"\n--- {tag}INPUT: Q va K (4 ptu dau moi token) ---")
    print("-" * 65)
    for t in range(TOKENS):
        print(f" [TOKEN {t}] Q: ", end="")
        for i in range(min(4, D_HEAD)):
            val = int(Q_int[t, i])
            print(f"{val & 0xFFFF:04x} ({val:^6}) | ", end="")
        print("")
        print(f" [TOKEN {t}] K: ", end="")
        for i in range(min(4, D_HEAD)):
            val = int(K_int[t, i])
            print(f"{val & 0xFFFF:04x} ({val:^6}) | ", end="")
        print("\n" + "-" * 65)

    print(f"\n--- {tag}PHASE 1: ATTENTION SCORE (Q x K^T / sqrt(d_head)) ---")
    print(f"    SQRT_SHIFT={SQRT_SHIFT}, divisor={256 * (1 << SQRT_SHIFT)}")
    print("-" * 58)
    for i in range(len(Score_flat)):
        val_int = int(Score_flat[i])
        print(f" [SCORE] [{i:^2}] {val_int & 0xFFFFFFFF:08x}  ({val_int:>6})  {val_int/256.0:.4f}")
    print("-" * 58)

    print(f"\n--- {tag}PHASE 2: EXP LUT ---")
    print("-" * 58)
    for i in range(len(exp_Z_flat)):
        val_int = int(exp_Z_flat[i])
        print(f" [EXP]   [{i:^2}]   {val_int & 0xFFFF:04x}        ({val_int:>4})  {val_int/256.0:.4f}")
    print("-" * 58)
    for t in range(TOKENS):
        val_sum = int(sum_exp_int[t, 0])
        print(f" >> SUM_EXP ROW {t}: {val_sum/256.0:.4f} (int: {val_sum})")

    print(f"\n--- {tag}PHASE 3: SOFTMAX WEIGHTS ---")
    print("-" * 58)
    for i in range(len(weights_int)):
        w_int = int(weights_int[i])
        print(f" [DIV]   [{i:^2}]   {w_int & 0xFFFF:04x}        ({w_int:>4})  {w_int/256.0:.4f}")
    print("-" * 58)

    return Score_int, exp_Z_int, sum_exp_int, weights_int_2d


# ==========================================
# 7. CHƯƠNG TRÌNH CHÍNH
# ==========================================
if __name__ == "__main__":

    # ── 0. NHẬP CẤU HÌNH TỪ BÀN PHÍM ───────────────────────────────────
    print("="*65)
    print("  CẤU HÌNH GOLDEN MODEL (nhấn Enter để giữ giá trị mặc định)")
    print("="*65)

    RUN_MODE = get_int_input(
        "Chọn RUN_MODE (1=Uniform, 2=Random)",
        RUN_MODE, min_val=1, max_val=2
    )

    TOKENS = get_int_input(
        "Nhập số ROW của ma trận Q/K/V (TOKENS)",
        TOKENS, min_val=1
    )
    D_HEAD = get_int_input(
        "Nhập số COL của ma trận Q/K/V (D_HEAD)",
        D_HEAD, min_val=1
    )

    # Cảnh báo: SQRT_SHIFT = log2(D_HEAD)//2 chỉ là phép xấp xỉ chính xác
    # khi D_HEAD là lũy thừa của 2 (giống cách qk_controller bên RTL shift
    # cứng bit). Nếu D_HEAD không phải lũy thừa của 2, golden score tính ra
    # vẫn chạy được nhưng KHÔNG còn khớp bit-true với phần cứng thực tế
    # trừ khi SQRT_SHIFT trong RTL cũng được tính lại tương ứng.
    if (D_HEAD & (D_HEAD - 1)) != 0:
        print(f"[CẢNH BÁO] D_HEAD = {D_HEAD} không phải lũy thừa của 2 "
              f"→ SQRT_SHIFT = log2(D_HEAD)//2 chỉ là giá trị xấp xỉ, "
              f"có thể KHÔNG khớp bit-true với RTL nếu RTL dùng shift cố định.")

    # ── A. Exp LUT (không đổi theo mode) ──────────────────────────────
    exp_lut_data = generate_exp_lut()
    write_coe_16("exp_rom.coe", exp_lut_data)
    write_mem_16("exp_rom.mem", exp_lut_data)

    # ── B. Generate Q, K, V theo mode ─────────────────────────────────
    if RUN_MODE == 1:
        print("\n" + "#"*65)
        print(f"  MODE 1: UNIFORM (tất cả phần tử = {UNIFORM_VAL})")
        print("#"*65)
        Q_int = np.full((TOKENS, D_HEAD), UNIFORM_VAL, dtype=np.int64)
        K_int = np.full((TOKENS, D_HEAD), UNIFORM_VAL, dtype=np.int64)
        V_int = np.full((TOKENS, D_HEAD), UNIFORM_VAL, dtype=np.int64)

    elif RUN_MODE == 2:
        print("\n" + "#"*65)
        print("  MODE 2: RANDOM (seed=42, range [-0.3, 0.3])")
        print("#"*65)
        np.random.seed(42)
        range_val = 0.3
        Q = np.random.uniform(-range_val, range_val, (TOKENS, D_HEAD)).astype(np.float32)
        K = np.random.uniform(-range_val, range_val, (TOKENS, D_HEAD)).astype(np.float32)
        V = np.random.uniform(-range_val, range_val, (TOKENS, D_HEAD)).astype(np.float32)
        Q_int = float_to_q88(Q)
        K_int = float_to_q88(K)
        V_int = float_to_q88(V)

    else:
        raise ValueError(f"RUN_MODE không hợp lệ: {RUN_MODE}. Chọn 1 hoặc 2.")

    Q_flat = Q_int.flatten()
    K_flat = K_int.flatten()
    V_flat = V_int.flatten()

    # ── C. Xuất file COE (cho BRAM initialization trong Vivado) ──────
    write_coe_32("q_ram.coe", Q_flat)
    write_coe_32("k_ram.coe", K_flat)
    write_coe_32("v_ram.coe", V_flat)

    # ── D. Xuất file MEM (cho TB $readmemh + S01_AXI load) ───────────
    write_mem_32("q_ram.mem", Q_flat)
    write_mem_32("k_ram.mem", K_flat)
    write_mem_32("v_ram.mem", V_flat)

    # ── E. Chạy Golden Model và xuất golden_score.mem ─────────────────
    label = "UNIFORM" if RUN_MODE == 1 else "RANDOM"
    Score_int, _, _, _ = run_golden(Q_int, K_int, V_int, exp_lut_data, label=label)
    write_golden_score(Score_int)

    print(f"\n[DONE] Mode {RUN_MODE} ({label}): tất cả file đã ghi vào {MEM_OUT_PATH}\n")