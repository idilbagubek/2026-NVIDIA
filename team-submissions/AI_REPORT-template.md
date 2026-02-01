# AI Agent Usage Report: QAOA-MTS Hybrid Optimization for LABS Problem

## 1. The Workflow

I used a **single AI agent workflow** with Claude as my primary development partner throughout the entire project:

```
[Problem Definition] → [Iterative Coding with Claude] → [Debug & Fix] → [Integration & Testing]
```

**Workflow Details:**
- **Architecture Design:** Discussed the hybrid QAOA + Memetic Tabu Search approach, Claude helped structure the algorithm flow
- **Coding:** Developed CUDA-Q kernels, custom quantum gates (RYZ, RZZZZ, etc.), and MTS algorithm iteratively
- **Debugging:** Pasted runtime errors directly to Claude for root cause analysis
- **Optimization:** Continuous refactoring for cleaner, more efficient code

The conversation was **highly iterative** - I would describe what I wanted in casual terms (sometimes in Turkish), and Claude would translate that into working code.

---

## 2. Verification Strategy

### Unit Tests for AI-Generated Code

**Test 1: Energy Function Validation**
```python
def test_energy_function():
    # Known LABS optimal for N=5: [+1, +1, +1, -1, +1] has E=2
    seq = np.array([1, 1, 1, -1, 1])
    assert energy(seq) == 2, f"Expected 2, got {energy(seq)}"
    
    # Symmetry test: flipping all spins shouldn't change energy
    seq_flipped = seq * -1
    assert energy(seq) == energy(seq_flipped), "Energy should be symmetric"
```

**Test 2: Bitstring to Spin Conversion**
```python
def test_bitstring_conversion():
    # "101" should become [1, -1, 1]
    result = bitstring_to_spin("101")
    expected = np.array([1, -1, 1])
    assert np.array_equal(result, expected), f"Got {result}"
    
    # Test padding
    result_padded = bitstring_to_spin("1", N=3)  # Should be "001" → [-1, -1, 1]
    assert len(result_padded) == 3
```

**Test 3: MTS Population Preservation Bug**
```python
def test_mts_preserves_best():
    # This test caught the critical bug where MTS was losing best solutions
    np.random.seed(42)
    pop = np.random.choice([-1, 1], size=(10, 20))
    initial_best = min(energy(p) for p in pop)
    
    best_seq, best_energy, _ = MTS(pop.copy(), max_iterations=100)
    
    # Best should never get worse
    assert best_energy <= initial_best, "MTS lost the best solution!"
```

---

## 3. The "Vibe" Log

### 🏆 Win: MTS Bug Detection Saved Hours

I was stuck for a long time - MTS kept getting stuck at the same energy value and never improving. I shared my code and output:

```
iter 50 best 75 child 107 same_child? False
iter 100 best 75 child 91 same_child? False
...
iter 500 best 75 child 155 same_child? False
```

Claude immediately identified the root cause: **the best solution wasn't being preserved in the population**. The fix was simple but I would have spent hours debugging this myself:

```python
# Bug: randomly overwriting could kill the best
population[random_idx] = child  # might overwrite best!

# Fix: replace worst member instead
worst_idx = np.argmax([energy(p) for p in population])
if child_energy < energy(population[worst_idx]):
    population[worst_idx] = child
```

**Time saved: ~3-4 hours** of debugging.

---

### 📋 Context Dump: Example Prompt

> "something is very wrong in this code but i dont know what: [code] output: [output showing stuck optimization]"

---

### ❌ Fail: CUDA-Q Kernel np.pi Hallucination

Claude generated quantum gate definitions using `np.pi`:

```python
@cudaq.kernel
def RYZ(theta: float, q0: cudaq.qubit, q1: cudaq.qubit):
    rx(np.pi/2,  q0)  # BUG: numpy doesn't work in CUDA-Q kernels!
    rz(theta, q0, q1)
    rx(-np.pi/2, q0)
```

This compiled initially but broke later with a cryptic error:
```
CompilerError: unhandled function call - RYZ, known kernels are dict_keys([])
```

**The Fix:** Replace `np.pi` with `math.pi`:

```python
import math

@cudaq.kernel
def RYZ(theta: float, q0: cudaq.qubit, q1: cudaq.qubit):
    rx(math.pi/2,  q0)  # math module works!
    rz(theta, q0, q1)
    rx(-math.pi/2, q0)
```

**Lesson:** AI doesn't always know framework-specific constraints. CUDA-Q kernels have restrictions on what Python features can be used inside them.

---

## Key Takeaways

1. **Iterative debugging** with AI is extremely effective - paste error, get fix
2. **Casual prompts** often work better than formal specifications
3. **Always verify** AI-generated code with unit tests - caught the MTS preservation bug
4. **Domain-specific constraints** (like CUDA-Q kernel limitations) need human knowledge

---

*This report was also written by an AI. 🤖*
