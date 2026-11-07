# nnms-ldpc-fpga
Fixed-point-aware neural min-sum decoding of short LDPC codes for low-power IoT receivers on a low-cost FPGA
# Quantized Neural Min-Sum LDPC Decoder — Python reference model

Step 2 of the paper *"A Quantized Neural Min-Sum LDPC Decoder for Short-Packet IoT Links:
Design and FPGA Implementation with In-Fabric Channel Emulation"*.

This folder is the algorithmic reference and the **bit-exact golden model** for the
Vivado HLS 2019.2 implementation on the ZedBoard (Zynq-7020).

## Code under study

5G NR base graph 2 (3GPP TS 38.212, Table 5.3.2-3), all 10 information columns
(K = 10·Z, no filler bits), first `m` base rows kept, first 2Z columns punctured.

| Setting (default) | Value |
|---|---|
| Lifting size Z | 16 (set index i_LS = 0) |
| Base rows m | 12 → rate R = 10/(8+m) = 1/2 |
| Info bits K | 160 |
| Transmitted bits N_tx | 320 (mother code N = 352) |
| Base-graph edges | 77 |
| Check-node degrees per layer | 8, 10, 8, 10, 4, 6, 6, 6, 4, 5, 5, 5 |

Other sizes: `--Z 24` gives K = 240; `--m 22` gives R ≈ 1/3. In the paper, call it a
"BG2-derived code" and state that rate matching (circular buffer, fillers) is omitted.

## Files

| File | Purpose |
|---|---|
| `ldpc_nms/code5g.py` | BG2 table loading, lifting, row pruning, GF(2) systematic encoder, layer description |
| `ldpc_nms/decoders.py` | Layered decoders in PyTorch: BP, MS, NMS, OMS, neural NMS; `FixedPointNMS` = integer golden model |
| `ldpc_nms/channel.py` | BPSK-AWGN with puncturing; Monte-Carlo BER/FER simulator |
| `train_neural.py` | Trains neural NMS weights (float, then quantization-aware to k/16); exports `weights_int.h` for HLS |
| `simulate.py` | BER/FER comparison of all decoders; writes CSV and plot |
| `ldpc_nms/data/5G_bg2.csv` | BG2 shift table (from NVIDIA Sionna, Apache-2.0; values from TS 38.212) |

## Run

```bash
pip install numpy scipy torch matplotlib
python train_neural.py --share edge   --out results/neural_edge      # ~15 min on 2 CPU cores
python train_neural.py --share layer  --steps 600 --qat_steps 150 --out results/neural_layer
python train_neural.py --share iter   --steps 400 --qat_steps 100 --out results/neural_iter
# Stage 3: fine-tune through the exact fixed-point arithmetic (bit-exact with the golden model)
python train_fixed_aware.py --round_mode floor --out results/neural_edge_fx_floor
python train_fixed_aware.py --share layer --init results/neural_layer/decoder.pt --out results/neural_layer_fx_floor
python train_fixed_aware.py --share iter  --init results/neural_iter/decoder.pt  --out results/neural_iter_fx_floor
python simulate.py --ebn0 1.0,1.5,2.0,2.5,3.0,3.5 --min_fe 200 --max_frames 500000 \
    --fx results/neural_edge_fx_floor:floor,results/neural_edge_fx_round:round --seed 2026 --out results/final
```

The weight table for HLS is `results/neural_edge_fx_floor/weights_int.h`. It also contains
the per-layer column indices and shift values.

## Training pipeline (3 stages)

1. **Float training** (`train_neural.py`): unrolled layered neural NMS, multi-loss BCE,
   Eb/N0 uniform in [1, 3] dB, 1500 steps × 500 frames, random codewords.
2. **Weight QAT** (same script): weights restricted to k/16 with a straight-through estimator.
3. **Fixed-point-aware fine-tuning** (`train_fixed_aware.py`): the forward pass
   reproduces the integer decoder exactly (input quantization, saturation, `(W·min)>>4`), with STE
   gradients. Its outputs are checked to be **bit-identical** to `FixedPointNMS`.

Stage 3 is essential. Float-trained weights lose their gain in fixed point because
truncation adds attenuation on top of the attenuation the weights already learned. After
stage 3, the 6/7/5-bit decoder with truncation matches floating-point BP from 2.5 dB upward.

## Main result (results/final, seed 2026, ≥200 frame errors or 500k frames per point)

Required Eb/N0 for the target frame error rate, BG2 Z=16, K=160, R=1/2, 10 layered iterations:

| Decoder | FER = 1e-2 | FER = 1e-3 |
|---|---|---|
| BP / sum-product (float) | 2.61 dB | 3.21 dB |
| MS (fixed 6/7/5) | ≈3.6 dB | > 3.5 dB |
| NMS 12/16 (fixed 6/7/5) | 2.76 dB | 3.34 dB |
| Neural NMS, float-trained weights (fixed 6/7/5) | 2.82 dB | 3.36 dB |
| **Neural NMS, fixed-point-aware, floor (fixed 6/7/5)** | **2.66 dB** | **3.21 dB** |

The proposed 6/7/5-bit decoder needs the same Eb/N0 as floating-point BP at FER = 1e-3. It
gains about 0.10–0.13 dB over fixed-point NMS and about 0.9 dB over fixed-point MS.

## Ablation: weight sharing (fixed point 6/7/5, floor, 10 iterations)

| Sharing | # weights | FER @ 3.0 dB | FER @ 3.5 dB |
|---|---|---|---|
| per edge, per iteration | 770 | 2.6e-3 | 3.6e-4 |
| per layer, per iteration | 120 | 3.2e-3 | 5.2e-4 |
| per iteration | 10 | 3.5e-3 | 5.4e-4 |
| fixed NMS 12/16 (no learning) | 1 | 3.7e-3 | 5.5e-4 |

## Design decisions that carry into hardware

* **Layered schedule, one base row per layer.** The Z checks of a layer are updated in
  parallel. This is the partially parallel architecture planned for the FPGA (Z processing units).
* **Normalized (multiplicative) weights restricted to k/16.** In hardware this is
  `(W·min) >> 4`, a 4-bit constant multiply, i.e. a few adders. Learned weights cost a
  small ROM of T × 77 entries × 5 bits.
* **Scale-invariant decoding.** MS/NMS/neural NMS are invariant to LLR scaling, so the
  fixed-point decoder quantizes the received sample y directly (scale 8, i.e. 3 fractional
  bits) and needs **no noise-variance estimate**. OMS is not scale-invariant, which is one
  argument for the normalized form.
* **Word lengths.** At Eb/N0 = 2.5 dB: channel 6 bits / posterior 7 bits / message 5 bits
  matches floating-point NMS. 5-bit channel input costs about 2× FER and 4-bit fails.
  Default design point: **6/7/5**.
* **Rounding.** `(W·min) >> 4` truncates (floor). Use the same in HLS for bit-exactness.

## Golden-model use for HLS verification

`FixedPointNMS.decode_int()` takes int32 channel values and returns int32 posteriors after
T iterations. Dump stimulus/response pairs from it to text files and compare them in the
HLS C testbench. Every value must match exactly.

## Step 3: HLS decoder core (`hls/`)

```bash
python export_hls.py                 # regenerates hls/nnms_tables.h and hls/golden/ (600 frames)
cd hls && vivado_hls -f run_hls.tcl  # Vivado HLS 2019.2: csim, csynth, cosim, IP export
```

C simulation status: **PASS**. All 352 posteriors of all 600 golden frames are bit-identical to
`FixedPointNMS`, with 10 iterations and with early termination (iteration counts match too).
It was checked with g++ and Xilinx's open-source ap_int headers. Synthesis has not b
