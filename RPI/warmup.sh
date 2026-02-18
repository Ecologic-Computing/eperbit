echo "Warming up CPU to scale frequency..."
WARMUP_SECS=15

for i in $(seq 1 $(nproc)); do
  yes > /dev/null &
  PIDS[$i]=$!
done

sleep "$WARMUP_SECS"

# Stop warmup
for pid in "${PIDS[@]}"; do
  kill "$pid" 2>/dev/null
done

wait 2>/dev/null

echo "Warmup done."

