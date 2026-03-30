"""
System-wide constants for SliceNet QoS Engine.
Mirrors real telecom and networking system parameters.
"""

# MTU (Maximum Transmission Unit) in bytes
MTU_BYTES = 1500
JUMBO_MTU_BYTES = 9000

# Default link capacities (Mbps)
LINK_CAPACITY_1G = 1_000
LINK_CAPACITY_10G = 10_000
LINK_CAPACITY_100G = 100_000

# TCP parameters
TCP_INITIAL_CWND = 10          # Initial congestion window (segments)
TCP_MAX_CWND = 65535           # Max congestion window
TCP_SSTHRESH_INIT = 65535      # Initial slow start threshold
TCP_MSS = 1460                 # Max segment size (bytes)
TCP_MIN_RTO_MS = 200           # Minimum retransmission timeout (ms)
TCP_MAX_RTO_MS = 120_000       # Maximum RTO (ms)
TCP_RETRANSMIT_LIMIT = 15      # Max retransmit attempts before drop

# RTT estimation (Jacobson/Karels algorithm)
RTT_ALPHA = 0.125              # SRTT smoothing factor
RTT_BETA = 0.25                # RTTVAR smoothing factor
RTT_INIT_MS = 100.0            # Initial RTT estimate (ms)

# Queue parameters
MAX_QUEUE_SIZE = 1000          # Packets per queue
RED_MIN_THRESHOLD = 200        # RED minimum threshold
RED_MAX_THRESHOLD = 600        # RED maximum threshold
RED_MAX_PROBABILITY = 0.1      # RED max drop probability
RED_WEIGHT = 0.002             # EWMA weight for avg queue length

# Token bucket defaults
TOKEN_BUCKET_MAX_BURST = 10_000  # bytes
TOKEN_BUCKET_RATE_BPS = 1_000_000  # 1 Mbps default

# Scheduling time quantum (microseconds)
DRR_QUANTUM_BYTES = 1500       # Default DRR quantum

# Slice IDs
SLICE_VOIP = "voip"
SLICE_VIDEO = "video"
SLICE_IOT = "iot"
SLICE_BEST_EFFORT = "best_effort"

# Priority levels (lower = higher priority)
PRIORITY_CRITICAL = 0
PRIORITY_HIGH = 1
PRIORITY_MEDIUM = 2
PRIORITY_LOW = 3
PRIORITY_BEST_EFFORT = 4

# Traffic class to slice mapping
TRAFFIC_SLICE_MAP = {
    "voip": SLICE_VOIP,
    "video": SLICE_VIDEO,
    "iot": SLICE_IOT,
    "http": SLICE_BEST_EFFORT,
    "bulk": SLICE_BEST_EFFORT,
}

# Slice to priority mapping
SLICE_PRIORITY_MAP = {
    SLICE_VOIP: PRIORITY_CRITICAL,
    SLICE_VIDEO: PRIORITY_HIGH,
    SLICE_IOT: PRIORITY_MEDIUM,
    SLICE_BEST_EFFORT: PRIORITY_BEST_EFFORT,
}

# WFQ weights (sum = 100)
SLICE_WEIGHT_MAP = {
    SLICE_VOIP: 40,
    SLICE_VIDEO: 35,
    SLICE_IOT: 15,
    SLICE_BEST_EFFORT: 10,
}

# Latency SLA targets (ms)
LATENCY_SLA_MS = {
    SLICE_VOIP: 20.0,
    SLICE_VIDEO: 50.0,
    SLICE_IOT: 100.0,
    SLICE_BEST_EFFORT: 500.0,
}

# Bandwidth guarantees (Mbps)
BANDWIDTH_GUARANTEE_MBPS = {
    SLICE_VOIP: 2.0,
    SLICE_VIDEO: 20.0,
    SLICE_IOT: 5.0,
    SLICE_BEST_EFFORT: 1.0,
}

# Simulation tick rate (ms)
SIMULATION_TICK_MS = 1.0

# Metrics window size (samples)
METRICS_WINDOW_SIZE = 1000
METRICS_PERCENTILES = [50, 95, 99]
