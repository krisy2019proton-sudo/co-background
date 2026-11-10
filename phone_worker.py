# qpy:service
import time

while True:
    print("Background script running...")
    time.sleep(5)

"""Phone worker: helps the PC train GRU models. Runs on an Android phone.

This file is self-contained: send ONLY this file to each phone (Pydroid 3).

Setup (once per phone):
  1. Set HOST to the PC's IP (the PC prints it when it starts).
  2. Set CLUSTER_PASSWORD to the same text as in the PC's script.
  3. Keep the phone plugged in and the screen on, then run this file.
The PC decides the model size, epochs and loss penalty; nothing to set here.
"""
import json
import math
import multiprocessing
import os
import random
import socket
import sys
import time
import zlib
from multiprocessing.connection import Client

# ==========================================
# SETTINGS
# ==========================================
HOST = "192.168.3.133"  # <- the PC's IP address on your WiFi
CLUSTER_PORT = 50000  # same as the PC's script
CLUSTER_PASSWORD = "rycbar"  # same text as the PC's script
PHONE_NAME = None  # None = random name like phone482, or set a fixed one
# None = automatic (about half the cores). 1 = single process.
# If extra processes don't work on the phone, it falls back to 1 by itself.
PROCESSES = None

# Replaced by the PC's values as soon as the phone connects:
HIDDEN_SIZE = 32
LEARNING_RATE = 0.008
EPOCHS = 30
LOSS_PENALTY = 2.0

CLUSTER_PROTOCOL = 1  # must match the PC's script
HEARTBEAT_SECONDS = 2.0  # how often a busy worker reports progress
RPC_TIMEOUT = 30  # seconds to wait for the PC before reconnecting


# ==========================================
# MATH + GRU (identical to the PC's script)
# ==========================================
def mat_vec_mul(A, x):
  return [sum(a * b for a, b in zip(row, x)) for row in A]


def mat_T_vec_mul(A, x):
  """Computes A^T @ x without building the transpose."""
  out = [0.0] * len(A[0])
  for i, row in enumerate(A):
    xi = x[i]
    for j, a in enumerate(row):
      out[j] += a * xi
  return out


def add_outer(M, a, b):
  """In-place M += outer(a, b)."""
  for i, ai in enumerate(a):
    row = M[i]
    for j, bj in enumerate(b):
      row[j] += ai * bj


def add_vec(v, d):
  """In-place v += d."""
  for i, di in enumerate(d):
    v[i] += di


def init_matrix(rows, cols):
  bound = math.sqrt(6.0 / (rows + cols))  # Xavier / Glorot uniform
  return [
      [random.uniform(-bound, bound) for _ in range(cols)]
      for _ in range(rows)
  ]


def zeros_like(p):
  if isinstance(p[0], list):
    return [[0.0] * len(row) for row in p]
  return [0.0] * len(p)


def sigmoid(x):
  if x >= 0:
    return 1.0 / (1.0 + math.exp(-x))
  e = math.exp(x)
  return e / (1.0 + e)


PARAM_NAMES = [
    "W_z", "U_z", "b_z",  # update gate
    "W_r", "U_r", "b_r",  # reset gate
    "W_n", "U_n", "b_n",  # candidate state
    "W_hy", "b_y",  # output layer
]


class GRU:
  """Single-layer GRU with a linear read-out on the last hidden state.

  z_t = sigmoid(W_z x_t + U_z h_{t-1} + b_z)          (update gate)
  r_t = sigmoid(W_r x_t + U_r h_{t-1} + b_r)          (reset gate)
  n_t = tanh(W_n x_t + U_n (r_t * h_{t-1}) + b_n)     (candidate state)
  h_t = (1 - z_t) * n_t + z_t * h_{t-1}
  """

  def __init__(self, input_size, hidden_size, output_size=1, lr=0.01):
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.lr = lr

    for gate in ("z", "r", "n"):
      setattr(self, f"W_{gate}", init_matrix(hidden_size, input_size))
      setattr(self, f"U_{gate}", init_matrix(hidden_size, hidden_size))
      setattr(self, f"b_{gate}", [0.0] * hidden_size)

    self.W_hy = init_matrix(output_size, hidden_size)
    self.b_y = [0.0]

  def forward(self, inputs):
    T = len(inputs)
    H = self.hidden_size
    h_states = {-1: [0.0] * H}
    steps = []

    for t in range(T):
      xt = inputs[t]
      h_prev = h_states[t - 1]

      Wz_x = mat_vec_mul(self.W_z, xt)
      Uz_h = mat_vec_mul(self.U_z, h_prev)
      z = [sigmoid(Wz_x[i] + Uz_h[i] + self.b_z[i]) for i in range(H)]

      Wr_x = mat_vec_mul(self.W_r, xt)
      Ur_h = mat_vec_mul(self.U_r, h_prev)
      r = [sigmoid(Wr_x[i] + Ur_h[i] + self.b_r[i]) for i in range(H)]

      rh = [r[i] * h_prev[i] for i in range(H)]
      Wn_x = mat_vec_mul(self.W_n, xt)
      Un_rh = mat_vec_mul(self.U_n, rh)
      n = [math.tanh(Wn_x[i] + Un_rh[i] + self.b_n[i]) for i in range(H)]

      h_t = [(1.0 - z[i]) * n[i] + z[i] * h_prev[i] for i in range(H)]
      h_states[t] = h_t
      steps.append({"z": z, "r": r, "n": n, "rh": rh})

    h_final = h_states[T - 1]
    pred_val = (
        sum(self.W_hy[0][i] * h_final[i] for i in range(H)) + self.b_y[0]
    )
    cache = {
        "inputs": inputs,
        "h_states": h_states,
        "steps": steps,
        "pred": pred_val,
    }
    return pred_val, cache

  @staticmethod
  def loss_weight(pred, target):
    """Extra weight for 'wasting money': predicted a gain, got a loss."""
    if pred > 0.0 and target < 0.0:
      return LOSS_PENALTY
    return 1.0

  def compute_gradients(self, cache, target):
    """BPTT for loss = w * 0.5 * (pred - target)^2, w = loss_weight()."""
    inputs = cache["inputs"]
    h_states = cache["h_states"]
    steps = cache["steps"]
    pred = cache["pred"]
    T = len(inputs)
    H = self.hidden_size

    grads = {name: zeros_like(getattr(self, name)) for name in PARAM_NAMES}

    d_pred = self.loss_weight(pred, target) * (pred - target)
    h_final = h_states[T - 1]
    grads["W_hy"] = [[d_pred * h_final[i] for i in range(H)]]
    grads["b_y"] = [d_pred]
    dh = [self.W_hy[0][i] * d_pred for i in range(H)]

    for t in range(T - 1, -1, -1):
      z, r, n, rh = (steps[t][k] for k in ("z", "r", "n", "rh"))
      h_prev = h_states[t - 1]
      xt = inputs[t]

      # h_t = (1 - z) * n + z * h_prev
      dn_raw = [dh[i] * (1.0 - z[i]) * (1.0 - n[i] ** 2) for i in range(H)]
      dz_raw = [
          dh[i] * (h_prev[i] - n[i]) * z[i] * (1.0 - z[i]) for i in range(H)
      ]

      # n_raw = W_n x + U_n (r * h_prev) + b_n
      d_rh = mat_T_vec_mul(self.U_n, dn_raw)
      dr_raw = [d_rh[i] * h_prev[i] * r[i] * (1.0 - r[i]) for i in range(H)]

      add_outer(grads["W_z"], dz_raw, xt)
      add_outer(grads["U_z"], dz_raw, h_prev)
      add_vec(grads["b_z"], dz_raw)

      add_outer(grads["W_r"], dr_raw, xt)
      add_outer(grads["U_r"], dr_raw, h_prev)
      add_vec(grads["b_r"], dr_raw)

      add_outer(grads["W_n"], dn_raw, xt)
      add_outer(grads["U_n"], dn_raw, rh)
      add_vec(grads["b_n"], dn_raw)

      if t > 0:
        # Gradient flowing back to h_{t-1} through every path
        dz_h = mat_T_vec_mul(self.U_z, dz_raw)
        dr_h = mat_T_vec_mul(self.U_r, dr_raw)
        dh = [
            dh[i] * z[i] + d_rh[i] * r[i] + dz_h[i] + dr_h[i]
            for i in range(H)
        ]

    return grads

  def apply_gradients(self, grads):
    lr = self.lr
    for name in PARAM_NAMES:
      p = getattr(self, name)
      g = grads[name]
      if isinstance(p[0], list):
        for i in range(len(p)):
          row, grow = p[i], g[i]
          for j in range(len(row)):
            row[j] -= lr * grow[j]
      else:
        for i in range(len(p)):
          p[i] -= lr * g[i]

  def backward(self, cache, target):
    self.apply_gradients(self.compute_gradients(cache, target))

  def fit(
      self, train_windows, epochs=30, label="", on_step=None, verbose=True,
      step_every=50,
  ):
    n = len(train_windows)
    done = 0
    for epoch in range(1, epochs + 1):
      random.shuffle(train_windows)
      total_loss = 0.0
      for sample in train_windows:
        pred, cache = self.forward(sample["inputs"])
        w = self.loss_weight(pred, sample["target"])
        total_loss += w * 0.5 * (pred - sample["target"]) ** 2
        self.backward(cache, sample["target"])
        done += 1
        if on_step is not None and done % step_every == 0:
          on_step(done)  # progress / heartbeat hook
      if verbose:
        print(
            f"    {label} epoch {epoch}/{epochs} - loss {total_loss / n:.8f}\n",
            end="",
            flush=True,
        )
    if on_step is not None:
      on_step(done)

  def predict(self, inputs):
    pred, _ = self.forward(inputs)
    return pred

  def save_model(self, filepath):
    model_data = {
        "input_size": self.input_size,
        "hidden_size": self.hidden_size,
        "output_size": self.output_size,
    }
    for name in PARAM_NAMES:
      model_data[name] = getattr(self, name)
    with open(filepath, "w", encoding="utf-8") as f:
      json.dump(model_data, f, indent=4)


# ==========================================
# WORKER
# ==========================================
def available_ram_mb():
  """Free RAM in MB (Linux), or None if it can't be determined."""
  try:
    with open("/proc/meminfo", encoding="utf-8") as f:
      for line in f:
        if line.startswith("MemAvailable:"):
          return int(line.split()[1]) / 1024.0
  except (OSError, ValueError, IndexError):
    pass
  return None


def format_duration(seconds):
  seconds = int(seconds)
  h, rem = divmod(seconds, 3600)
  m, sec = divmod(rem, 60)
  if h:
    return f"{h}h{m:02d}m"
  if m:
    return f"{m}m{sec:02d}s"
  return f"{sec}s"


def _authkey():
  return CLUSTER_PASSWORD.encode("utf-8")


class _BadPassword(Exception):
  pass


class _Cancelled(Exception):
  pass


class _Link:
  """Worker side of the connection, with automatic reconnect."""

  def __init__(self, host, port, say):
    self.host, self.port, self.say = host, port, say
    self.conn = None

  def _drop(self):
    if self.conn is not None:
      try:
        self.conn.close()
      except OSError:
        pass
      self.conn = None

  def _open(self):
    try:
      self.conn = Client((self.host, self.port), authkey=_authkey())
    except multiprocessing.AuthenticationError:
      raise _BadPassword()

  def wait_for_coordinator(self):
    while True:
      try:
        self._open()
        return
      except (OSError, EOFError):
        self.say(f"[WORKER] Waiting for the PC at {self.host}:{self.port} ...")
        time.sleep(3)

  def call(self, name, *args, tries=5):
    for attempt in range(tries):
      try:
        if self.conn is None:
          self._open()
        self.conn.send((name, args))
        if not self.conn.poll(RPC_TIMEOUT):
          raise OSError("coordinator did not answer")
        status, value = self.conn.recv()
      except (OSError, EOFError):
        self._drop()
        if attempt < tries - 1:
          time.sleep(2)
        continue
      if status != "ok":
        raise RuntimeError(value)
      return value
    raise ConnectionError("coordinator unreachable")


def _train_job(link, worker_id, ticker, attempt, windows, say):
  random.seed(zlib.crc32(ticker.encode("utf-8")))
  model = GRU(
      input_size=3, hidden_size=HIDDEN_SIZE, output_size=1, lr=LEARNING_RATE
  )
  total = EPOCHS * len(windows)
  last = {"beat": 0.0, "decile": -1}

  def on_step(done):
    pct = int(100 * done / total)
    if pct // 10 != last["decile"]:
      last["decile"] = pct // 10
      say(f"[WORKER] {ticker}: {pct}%")
    now = time.time()
    if now - last["beat"] < HEARTBEAT_SECONDS:
      return
    last["beat"] = now
    try:
      alive = link.call(
          "heartbeat", worker_id, ticker, attempt, done, tries=2
      )
    except ConnectionError:
      return  # network hiccup: keep training
    if not alive:
      raise _Cancelled()

  # Adaptive step frequency: updates progress roughly every 5% of training
  step_every = max(1, total // 20)

  model.fit(
      windows, epochs=EPOCHS, label=ticker, on_step=on_step, verbose=False,
      step_every=step_every,
  )
  return {name: getattr(model, name) for name in PARAM_NAMES}


def worker_loop(host, port, label="worker", quiet=False):
  """Pulls tickers from the PC and trains them. Used by PC cores and phones."""
  def _say(*args):
    if not quiet:
      print(*args, flush=True)
      sys.stdout.flush()
  say = _say

  try:
    hostname = socket.gethostname()
  except OSError:
    hostname = "device"
  label = label.replace("-", "_")  # '-' separates the parts of the id
  worker_id = f"{label}-{hostname}-{os.getpid()}"
  link = _Link(host, port, say)
  say(f"[WORKER] {label} starting; PC is {host}:{port}")
  try:
    link.wait_for_coordinator()
    config = link.call("get_config")
    if (
        config.get("PROTOCOL") != CLUSTER_PROTOCOL
        or config.get("PARAM_NAMES") != PARAM_NAMES
    ):
      say(
          "[WORKER] Version mismatch: this script and the PC's script are out"
          " of sync. Copy the newest phone_worker.py from the PC's folder."
      )
      return
    for key in ("HIDDEN_SIZE", "LEARNING_RATE", "EPOCHS", "LOSS_PENALTY"):
      globals()[key] = config[key]  # use the PC's hyperparameters
    say("[WORKER] Connected. Waiting for jobs ...")
    while True:
      try:
        job = link.call("claim_job", worker_id)
      except ConnectionError:
        say("[WORKER] The PC is gone. Exiting.")
        return
      if job == "FINISHED":
        say("[WORKER] All tickers are done. Exiting.")
        return
      if job is None:
        time.sleep(2)
        continue
      ticker, attempt, windows = job
      say(f"[WORKER] Training {ticker} ({len(windows)} samples) ...")
      started = time.time()
      error = None
      params = None
      try:
        params = _train_job(link, worker_id, ticker, attempt, windows, say)
      except _Cancelled:
        say(f"[WORKER] {ticker}: no longer needed, skipping.")
        continue
      except Exception as e:  # report it to the PC instead of dying
        error = f"{type(e).__name__}: {e}"
      try:
        link.call("submit", worker_id, ticker, attempt, params, error)
        elapsed = format_duration(time.time() - started)
        say(f"[WORKER] {ticker} delivered ({elapsed}).")
      except ConnectionError:
        say(f"[WORKER] Could not deliver {ticker}; the PC will reassign it.")
  except _BadPassword:
    say("[WORKER] Wrong password: CLUSTER_PASSWORD must match the PC's.")
  except KeyboardInterrupt:
    say("[WORKER] Stopped.")


def auto_phone_processes():
  """Processes for one phone: about half its cores, so it stays cool."""
  try:
    cores = len(os.sched_getaffinity(0))
  except AttributeError:
    cores = os.cpu_count() or 1
  by_cpu = min(4, max(1, cores // 2))
  free_mb = available_ram_mb()
  by_ram = by_cpu if free_mb is None else int(0.6 * free_mb / 150)  # ~150 MB each
  return max(1, min(by_cpu, by_ram)), cores


def run_phone(host, port, label="phone", processes=None):
  """Phone entry point: tries several processes, falls back to a single one."""
  if processes is None:
    processes, cores = auto_phone_processes()
    print(f"[PHONE] auto: {cores} cores -> trying {processes} process(es)")
  helpers = []
  if processes > 1:
    try:
      for _ in range(processes - 1):
        p = multiprocessing.Process(
            target=worker_loop, args=(host, port, label, True), daemon=True
        )
        p.start()
        helpers.append(p)
      time.sleep(2)  # give a broken helper time to die
      running = [p for p in helpers if p.is_alive()]
      if not running:
        raise RuntimeError(f"helpers exited at once (code {helpers[0].exitcode})")
      if len(running) < len(helpers):
        print(f"[PHONE] Only {len(running)} of {len(helpers)} helpers started.")
      helpers = running
    except Exception as e:  # multiprocessing not supported on this phone
      print(
          f"[PHONE] Extra processes failed ({type(e).__name__}: {e}); using a"
          " single process."
      )
      for p in helpers:
        try:
          p.terminate()
        except Exception:  # pylint: disable=broad-except
          pass
      helpers = []
  print(f"[PHONE] Running {len(helpers) + 1} process(es).", flush=True)
  worker_loop(host, port, label)
  for p in helpers:
    p.join(timeout=5)


if __name__ == "__main__":
  name = PHONE_NAME or f"phone{random.randint(100, 999)}"
  print(f"[PHONE] {name}; PC is {HOST}:{CLUSTER_PORT}")
  run_phone(HOST, CLUSTER_PORT, name, PROCESSES)
