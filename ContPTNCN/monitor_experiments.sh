#!/bin/bash
# Monitor parallel hyperparameter tuning experiments

echo "================================================================"
echo "Monitoring Hyperparameter Tuning Experiments"
echo "================================================================"
echo ""

# Find the latest log files
VANILLA_LOG=$(ls -t results/vanilla_rnn_*.log 2>/dev/null | head -1)
FAST_LOG=$(ls -t results/fast_weights_*.log 2>/dev/null | head -1)

if [ -z "$VANILLA_LOG" ] || [ -z "$FAST_LOG" ]; then
    echo "Error: Log files not found!"
    echo "Looking for:"
    echo "  results/vanilla_rnn_*.log"
    echo "  results/fast_weights_*.log"
    exit 1
fi

echo "Monitoring logs:"
echo "  Vanilla RNN:    $VANILLA_LOG"
echo "  Fast Weights:   $FAST_LOG"
echo ""

# Check if processes are running
VANILLA_PID=$(pgrep -f "vanilla_rnn_tuning.json")
FAST_PID=$(pgrep -f "fast_weights_tuning.json")

if [ -n "$VANILLA_PID" ]; then
    echo "✓ Vanilla RNN experiment running (PID: $VANILLA_PID)"
else
    echo "✗ Vanilla RNN experiment not running"
fi

if [ -n "$FAST_PID" ]; then
    echo "✓ Fast Weights experiment running (PID: $FAST_PID)"
else
    echo "✗ Fast Weights experiment not running"
fi

echo ""
echo "================================================================"
echo "Recent Progress"
echo "================================================================"
echo ""

echo "--- Vanilla RNN (last 20 lines) ---"
tail -20 "$VANILLA_LOG" | grep -E "(Trial|Epoch|BPC|Error)" || echo "No recent updates"
echo ""

echo "--- Fast Weights (last 20 lines) ---"
tail -20 "$FAST_LOG" | grep -E "(Trial|Epoch|BPC|Error)" || echo "No recent updates"
echo ""

echo "================================================================"
echo "To follow logs in real-time:"
echo "  tail -f $VANILLA_LOG"
echo "  tail -f $FAST_LOG"
echo "================================================================"
