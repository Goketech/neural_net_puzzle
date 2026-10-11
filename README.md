# Neural Net Puzzle

A solver for a "reassemble the shuffled neural network" puzzle. A trained network was split into 97 separate layer files, and the job is to recover the order they were originally stacked in.

## The puzzle

`pieces/` holds 97 PyTorch state dicts (`piece_0.pth` … `piece_96.pth`), each containing a `weight` and `bias`. Together they form:

- **48 residual blocks**, each built from two linear layers:
  - an input layer `48 → 96` (weight shape `(96, 48)`)
  - an output layer `96 → 48` (weight shape `(48, 96)`)
  - computing `x + out(ReLU(inp(x)))`
- **1 final layer** `48 → 1` (weight shape `(1, 48)`)

`historical_data.csv` has 10,000 rows of inputs (`measurement_0` … `measurement_47`), the original model's output (`pred`), and the ground-truth target (`true`).

The answer is a permutation of the piece indices: for each block in order, its input layer then its output layer, followed by the final layer.

## How the solver works

`solve.py` runs in four stages:

1. **Sort pieces by shape.** Weight shapes separate the 48 input layers, 48 output layers and the single final layer.
2. **Pair the layers into blocks.** For every candidate pair, it scores the mean of the diagonal of `W_out @ W_in`. Layers that were trained together tend to give a strongly negative diagonal. The Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) then finds the pairing with the lowest total score.
3. **Make a first guess at the order.** Each block is run on the data and ranked by how much it changes its input (mean `‖block(x) − x‖`), smallest first.
4. **Refine with local search.** It tries swapping every pair of blocks and keeps any swap that lowers the MSE between the network's output and the `pred` column. This repeats until no swap helps or the MSE falls below `1e-6`.

At the end it prints the permutation as a comma-separated list.

## Usage

Requires Python 3 with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch numpy pandas scipy
```

Then run:

```bash
python solve.py
```

Example output:

```
Loading pieces...
Found 48 inputs, 48 outputs, and final layer at index ...
Matching blocks via W_out @ W_in negative diagonals...
Loading historical_data.csv...
Initial Sorted MSE: ...
Beginning local search to find perfect sequence...
Found better sequence! Swapped i & j. New MSE: ...

--- FINAL PERMUTATION SOLUTION ---
<97 comma-separated piece indices>
```

## Project layout

```
.
├── historical_data.csv   # inputs, original predictions, and targets
├── pieces/               # 97 shuffled layer state dicts
└── solve.py              # the solver
```
