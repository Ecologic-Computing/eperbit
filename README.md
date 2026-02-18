# eperbit

![GitHub License](https://img.shields.io/github/license/Ecologic-Computing/eperbit?style=flat&color=22c55e) [![DOI](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.18684348-blue)](https://doi.org/10.5281/zenodo.18684348)  [![PyPI - Package Name](https://img.shields.io/pypi/v/ecidcodes?label=ecidcodes)](https://pypi.org/project/ecidcodes/) [![PyPI Downloads](https://img.shields.io/pypi/dm/ecidcodes)](https://pypi.org/project/ecidcodes/) 
[![Last Updated](https://img.shields.io/github/last-commit/Ecologic-Computing/eperbit)](https://github.com/Ecologic-Computing/eperbit/commits)

Energy per bit measurements for ID codes.
This repository contains the code, scripts, and measurement data used to evaluate the **latency and energy consumption** of several identification (ID) codes and hash-based identification schemes on a testbed consisting of a Raspberry Pi 5 and Keysight DC Power Analyzer.


## **What’s Being Evaluated**

We benchmark the following ID code families:

* RSID (Reed–Solomon ID)

* RS2ID (Concatenated Reed–Solomon ID)

* RMID (Reed–Muller ID)

* SHA256ID (SHA-2 256 based identification code)

* PMHID (Polymur Hash–based ID)

Each code is implemented over Galois Field (GF) symbol sizes:

* 8

* 16

* 32

* 64

The goal is to compare their:

* Encoding latency

* Power and Energy consumption

* Energy cost of a bit of information encoded 

All tests are run repeatedly (typically \~20 seconds or ≥10,000 iterations) to clearly separate active processing from background noise.



## **Testbed Overview**

All measurements were performed on:

* **Device Under Test (DUT):** 64-bit Raspberry Pi 5

  * BCM2712, Cortex-A76

  * 8GB RAM

* **OS:** Debian 12

* **Kernel:** 6.12.47

* **Compiler:** GCC 12.2.0
* **IDCODES Library:**
    ```bash
    pip install ecidcodes
    ```
**Compile Flags:**

 `-O3 -mtune=cortex-a76 -march=armv8-a+crypto`


* **OpenSSL Version:** 3.0.18

### **Power Measurement Setup**

* **Power Analyzer:** Keysight N6705B DC Power Analyzer

  * N6785A module

* **Voltage supply:** 5.2V via Source to GPIO on Rpi 5

* **Data Logger Sampling Period:** 0.1 ms

We record `.dlog` files and post-process them to extract:

* Time

* Voltage

* Current

Background noise was minimized:

* WiFi and Bluetooth disabled

* Ethernet physically removed during tests

* LEDs disabled after boot

* Passive cooling (heatsink only)

* Controlled room temperature 

Warmup and sleep phases are added to stabilized temperature and CPU frequency and allow us to programmatically separate active compute phases from background fluctuations.

---

## **Repository Structure**

`.`  
`├── DATASET20260212/`  
`├── RPI/`  
`├── eperbit.py`  
`├── parselogs.py`  
`├── finalsummary.csv`  
`└── ...`

### **Top-Level Files**

* `eperbit.py` — energy per bit computation and metrics evaluation

* `parselogs.py` — parsing log files including Time and Temperature

* `finalsummary.csv` — final aggregated benchmark results from `.log` files

* `version.txt` — experiment version reference

* `LICENSE` / `COPYING` — licensing information

### **`DATASET20260212/`**

Contains:

* Measurement logs and labeled segments for:

  * PMHID-{8,16,32,64}

  * RMID-{8,16,32,64}

  * RSID-{8,16,32,64}

  * RS2ID-{16,32,64}

  * SHA256ID-{8,16,32,64}

* Log processing and segmentation scripts in `DATASET20260212`:

  * `dataset.py`

  * `segmentify.py`

  * `segmentify_all.sh`

  * `parselogs.py`

  * `dlogviewer.py`

  * `exportdlog_to_csv.sh`

  * `msgsizeveclen.sh`

  * `sortfiles.sh`

* Aggregated data:

  * `combined_summarylogs.csv`

These scripts convert raw Keysight `.dlog` files into structured CSV data and labeled segments for energy analysis.

---

### **`RPI/`**

Scripts used directly on the Raspberry Pi 5 during benchmarking:

* `warmup.sh` — CPU warmup phase

* `test.sh` — run benchmark

* `time.sh` — Set System Clock Time

* `killservices.sh` — disable background services

* `screen_test.sh` — run `test.sh` inside screen session

* `stress_test_results.txt` — stress test output
```
# Configure RPI to allow boot and power over GPIO and prevent hard crashes
sudo rpi-eeprom-config --edit
PSU_MAX_CURRENT=5000 
```


## **How the Benchmarks Work?**

Each benchmark run:

1. Generates a random input message (seeded by date/time).

2. Uses a fixed tag position for consistent energy comparison.

3. Executes the encoding algorithm repeatedly.

4. Inserts warmup and sleep phases.

5. Records high-resolution voltage and current traces.

6. Segments and labels active compute regions.

7. Computes:

   * Average latency

   * Energy consumption

   * Energy per bit

The repetition ensures that the encoding phase stands out clearly against  Linux background process activity.

---

## **Why Energy per bit  Matters?**

This repository provides:

* Real measured power traces with 0.1ms accuracy

* Repeatable benchmarking scripts

* Segmented dataset ready for analysis including timestamped periods of activity during benchmark and idle state

* Direct comparison across GF sizes and ID constructions

---

## **Reproducing the Experiments**

1. Set up a Raspberry Pi 5 with Debian 12 (64-bit).

2. Disable background services, radios, and LEDs using `killservices.sh`.

3. Use a high-resolution DC power analyzer and connect via USB-C or GPIO 5V.

4. Record `.dlog` files and convert to CSV using `dlogviewer.py`.

5. Process logs using scripts in `DATASET20260212/`.

6. Analyze aggregated results via `eperbit.py`.

---

## **License**

See `LICENSE` and `COPYING` for full licensing details.

Copyright (c) 2026 Ecologic Computing GmbH <info@ecologic-computing.com>

Visit [ecologic-computing.com/research](https://www.ecologic-computing.com/research) for more details.
