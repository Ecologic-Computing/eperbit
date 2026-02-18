#!/usr/bin/env bash
set -e
#set -ex
echo "Test BEGIN AT:::===== $(date '+%Y-%m-%d %H:%M:%S') ====="
echo "CPU frequency at start: $(vcgencmd measure_clock arm 2>/dev/null || echo N/A)"
# Run Warmup
bash ./warmup.sh
echo "CPU frequency after warmup: $(vcgencmd measure_clock arm 2>/dev/null || echo N/A)"
get_temp_sys() {
    awk '{printf "%.1f\n", $1/1000}' /sys/class/thermal/thermal_zone0/temp
}
# Fixed parameters
#GF2_EXPS=(8 16 32 64)
GF2_EXPS=( 64 )
ID_CODE_TYPES=("PMHID")
if [[ -n "$1" ]]; then
    GF2_EXPS=("$1")
fi

if [[ -n "$2" ]]; then
    ID_CODE_TYPES=("$2")
fi

TAG_POS=2
RS2_INNER_TAG_POS=2
N_ITER=10000
BENCHMARKSCRIPT="./benchmark_latest"
# BENCHMARKSCRIPT="./benchmark_latest_fixedtime"
RM_ORDER=1
CSV_PATH="./"
DATA_TYPE="random"
INPUT_DATA="HelloWorld"
VEC_LENS=(32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152)
# Redefine N_ITER based on VEC_LEN
# if ((vec_len < 1025 )); then
# N_ITER=20000000
# else
# N_ITER=20000000
# fi

# Assuming list of one element
key="${ID_CODE_TYPES[0]}-${GF2_EXPS[0]}"
#                                     32         64        128        256        512       1024      2048      4096      8192     16384    32768    65536   131072   262144  524288  1048576  2097152
case "$key" in
    PMHID-8)      N_ITER_LIST=(386922035  267630135  215100021  160307790  104536901   63869195   35055124   18271514   9259730   4827559   2418066  1209343   605479   301458   151433    75482    37878    ) ;;
    PMHID-16)     N_ITER_LIST=(431965442  386922035  258799171  212201591  162879713  105574324   59844404   34766283  18268009   9416062   4802762  2403799  1216545   605424   302960   151471    75475    ) ;;
    PMHID-32)     N_ITER_LIST=(443754160  443557329  386922035  268817204  213583938  162879713  103847551   63867156  34318267  18280031   9257030  4700717  2418189  1182489   599454   299805   150916    ) ;;
    PMHID-64)     N_ITER_LIST=(520697734  444247001  413564929  386922035  271665308  209819555  158692374  105362975  63526347  34235437  18356539  9506064  4699700  2417315  1183219   602982   302933    ) ;;
    RMID-8)       N_ITER_LIST=( 25716526   20927945   15700312   10441792    6290712    3459854    1827462     935570    474837    238309    119559    60634    29694    15180    10000    10000    10000    ) ;;
    RMID-16)      N_ITER_LIST=( 29692092   27656396   23380327   18291065   12828654    8172572    4713290    2540492   1315644    662518    332716   167325    83032    41675    20958    10287    10000    ) ;;
    RMID-32)      N_ITER_LIST=( 22542577   22722108   22656214   20862236   17727196   13731548    9267539    5728197   3324760   1777891    924946   470293   234916   117656    58793    29355    14577    ) ;;
    RMID-64)      N_ITER_LIST=( 15895471   15914443   15880325   15780711   15569654   14178363   11870563    8919452   5993748   3603227   2005129  1058269   537369   275542   137864    68615    34020    ) ;;
    RS2ID-16)     N_ITER_LIST=( 64217826   42122999   24869744   13622029    6900882    3265300    1605686     804482    407030    203348    101327    50239    25134    12328    10000    10000    10000    ) ;;
    RS2ID-32)     N_ITER_LIST=( 79248722   50327126   31346488   15299643    7095519    3208377    1531427     607951    255954    112328     52145    24947    12136    10000    10000    10000    10000    ) ;;
    RS2ID-64)     N_ITER_LIST=( 80035215   59150597   37132619   20078103   10153933    4807160    1967022     748662    292554    156831     66644    32202    15221    10000    10000    10000    10000    ) ;;
    RSID-8)       N_ITER_LIST=( 53330489   29330243   15229276    7795265    3947396    1978315     994002     497531    248631    124287     62160    31073    15535    10000    10000    10000    10000    ) ;;
    RSID-16)      N_ITER_LIST=( 89281728   51906257   28151568   14475341    7089508    3336035    1625732     815949    407758    203853    100900    50188    25092    12526    10000    10000    10000    ) ;;
    RSID-32)      N_ITER_LIST=(120503705   65218809   37360830   16754768    7165633    3349606    1481614     601096    258892    112167     52202    24895    12157    10000    10000    10000    10000    ) ;;
    RSID-64)      N_ITER_LIST=(158528852   91074681   46218196   24066230   11033387    4879738    1938022     746435    291911    156827     66813    32168    15184    10000    10000    10000    10000    ) ;;
    SHA256ID-8)   N_ITER_LIST=(  4547356    4574868    4455126    4359644    4343133    4054352    3415895    2813006   2037803   1311331    770722   422612   222251   114052    57758    29067    14573    ) ;;
    SHA256ID-16)  N_ITER_LIST=(  4636079    4626160    4539955    4380163    4182507    3918395    3343464    2773348   2026375   1317922    770578   423470   222195   114024    57747    29060    14575    ) ;;
    SHA256ID-32)  N_ITER_LIST=(  4623732    4483993    4567555    4423985    4240072    4000720    3404446    2779487   2011710   1309985    771823   422221   222173   113898    57741    29064    14569    ) ;;
    SHA256ID-64)  N_ITER_LIST=(  4397827    4531591    4076923    4129833    4184363    3982857    3292284    2697639   2007197   1302219    768788   422042   222062   113920    57726    29063    14577    ) ;;

    *) echo "No match found"
    exit 1
        ;;
esac

#VEC_LENS=(32 64 128 256 512 1024)
# ID_CODE_TYPES=("RSID" "RS2ID" "RMID" "PMHID" "SHA1ID" "SHA256ID")
TS=$(date '+%Y%m%d_%H%M%S')
mkdir -p "./logs"
for gf2_exp in "${GF2_EXPS[@]}"; do
    sleep 5
    for id_code in "${ID_CODE_TYPES[@]}"; do
        #sleep 5
        for veclenidx in "${!VEC_LENS[@]}";  do
        vec_len="${VEC_LENS[$veclenidx]}"
        # Use a customized N_ITER to cover atleast 20s or 10k iter.
        N_ITER="${N_ITER_LIST[$veclenidx]}"
        sleep 2
        {
        echo "System Temperature at START: $(get_temp_sys) °C"
        echo "START:::===== $(date '+%Y-%m-%d %H:%M:%S') ====="
        echo "Running ${id_code}-${gf2_exp}, vec_len=${vec_len},Iter=${N_ITER}"
        echo "========================================"
        sudo nice -n -19 taskset -c 1 $BENCHMARKSCRIPT \
        --gf2_exp "${gf2_exp}" \
        --tag_pos "${TAG_POS}" \
        --input_data "${INPUT_DATA}" \
        --rs2_inner_tag_pos "${RS2_INNER_TAG_POS}" \
        --id_code_type "${id_code}" \
        --n_iter "${N_ITER}" \
        --vec_len "${vec_len}" \
        --rm_order "${RM_ORDER}" \
        --csv_file_path "${CSV_PATH}" \
        --data_type "${DATA_TYPE}" \
        --force 
        #2>&1 | tee -a "./logs/${id_code}-${GF2_EXP}-${vec_len}-${n_iter}-${n_iter}-${TS}.log"
        echo "System Temperature at END: $(get_temp_sys) °C"
        echo "Finished Benchmark: ${id_code}-${GF2_EXP}, vec_len=${vec_len}., Iter=${N_ITER}"
        echo "END:::===== $(date '+%Y-%m-%d %H:%M:%S') ====="
        }  2>&1 | tee -a "./logs/${id_code}-${gf2_exp}-${vec_len}-${N_ITER}-${DATA_TYPE}-${TS}.log"
        sleep 2
        done
    done
done

echo "Test FINISHED AT:::===== $(date '+%Y-%m-%d %H:%M:%S') ====="
