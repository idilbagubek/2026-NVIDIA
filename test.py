import numpy as np
import unittest

def Combine(p1, p2):
    n = len(p1)
    k = np.random.randint(1, n)
    return np.concatenate([p1[:k], p2[k:]])

def Mutate(seq, p_mut):
    seq = seq.copy()
    for i in range(len(seq)):
        if (np.random.random() < p_mut):
            seq[i] *= -1
    return seq

def flip(seq, i):
    s_new = seq.copy()
    s_new[i] *= -1
    return s_new

def energy(seq):
    E = 0
    n = len(seq)
    for k in range(1, n):
        C_k = np.sum(seq[:n-k] * seq[k:])
        E += C_k * C_k
    return E

def get_interactions(N):
    
    G2 = []
    G4 = []

    for i in range(N-2):
        for k in range(1, (N-i) // 2):
            G2.append([i, i + k])

    for i in range(N-3):
        for t in range(1, (N-i-1) // 2):
            for k in range(t + 1, N-i-t):
                G4.append([i, i + t, i + k, i + k + t])

    return G2, G4

def flatten_G2(G2):
    G2_i = [int(i) for (i, j) in G2]
    G2_j = [int(j) for (i, j) in G2]
    return G2_i, G2_j

def flatten_G4(G4):
    a = [int(x[0]) for x in G4]
    b = [int(x[1]) for x in G4]
    c = [int(x[2]) for x in G4]
    d = [int(x[3]) for x in G4]
    return a, b, c, d

def bitstring_to_spin(b: str, N: int) -> np.ndarray:
    b = b.replace(" ", "").zfill(N)
    bits = b[-N:]
    return np.array([1 if ch == "1" else -1 for ch in bits], dtype=np.int8)

class TestEnergyFunction(unittest.TestCase):
    """Tests for the LABS energy function"""
    
    def test_known_optimal_n5(self):
        """Test known optimal sequence for N=5"""
        # Known LABS optimal for N=5: E=2
        seq = np.array([1, 1, 1, -1, 1])
        self.assertEqual(energy(seq), 2)
    
    def test_energy_symmetry(self):
        """Flipping all spins should not change energy"""
        seq = np.array([1, 1, -1, 1, -1, -1, 1])
        seq_flipped = seq * -1
        self.assertEqual(energy(seq), energy(seq_flipped))
    
    def test_energy_reversal(self):
        """Reversing the sequence should not change energy"""
        seq = np.array([1, 1, -1, 1, -1, -1, 1, 1])
        seq_reversed = seq[::-1]
        self.assertEqual(energy(seq), energy(seq_reversed))
    
    def test_energy_non_negative(self):
        """Energy should always be non-negative"""
        for _ in range(100):
            seq = np.random.choice([-1, 1], size=20)
            self.assertGreaterEqual(energy(seq), 0)
    
    def test_all_same_sequence(self):
        """All +1 or all -1 sequence has maximum autocorrelation"""
        seq_all_plus = np.ones(10, dtype=int)
        seq_all_minus = -np.ones(10, dtype=int)
        # Energy should be the same for both
        self.assertEqual(energy(seq_all_plus), energy(seq_all_minus))
        # And it should be quite high
        self.assertGreater(energy(seq_all_plus), 0)


class TestFlipFunction(unittest.TestCase):
    """Tests for the flip function"""
    
    def test_flip_changes_one_element(self):
        """Flip should only change one element"""
        seq = np.array([1, 1, -1, 1, -1])
        for i in range(len(seq)):
            flipped = flip(seq, i)
            # Check only index i changed
            for j in range(len(seq)):
                if j == i:
                    self.assertEqual(flipped[j], -seq[j])
                else:
                    self.assertEqual(flipped[j], seq[j])
    
    def test_flip_does_not_modify_original(self):
        """Flip should not modify the original sequence"""
        seq = np.array([1, 1, -1, 1, -1])
        original = seq.copy()
        _ = flip(seq, 2)
        np.testing.assert_array_equal(seq, original)
    
    def test_double_flip_returns_original(self):
        """Flipping twice at same index returns original"""
        seq = np.array([1, 1, -1, 1, -1])
        flipped_once = flip(seq, 2)
        flipped_twice = flip(flipped_once, 2)
        np.testing.assert_array_equal(seq, flipped_twice)


class TestCombineFunction(unittest.TestCase):
    """Tests for the Combine (crossover) function"""
    
    def test_combine_length(self):
        """Combined sequence should have same length as parents"""
        p1 = np.array([1, 1, 1, 1, 1])
        p2 = np.array([-1, -1, -1, -1, -1])
        child = Combine(p1, p2)
        self.assertEqual(len(child), len(p1))
    
    def test_combine_values(self):
        """Combined sequence should only contain values from parents"""
        p1 = np.array([1, 1, 1, 1, 1])
        p2 = np.array([-1, -1, -1, -1, -1])
        for _ in range(50):
            child = Combine(p1, p2)
            for val in child:
                self.assertIn(val, [-1, 1])
    
    def test_combine_crossover_structure(self):
        """Child should have prefix from p1 and suffix from p2"""
        np.random.seed(42)
        p1 = np.array([1, 1, 1, 1, 1])
        p2 = np.array([-1, -1, -1, -1, -1])
        child = Combine(p1, p2)
        # Find crossover point
        k = 0
        for i in range(len(child)):
            if child[i] == 1:
                k = i + 1
            else:
                break
        # Verify structure
        np.testing.assert_array_equal(child[:k], p1[:k])
        np.testing.assert_array_equal(child[k:], p2[k:])


class TestMutateFunction(unittest.TestCase):
    """Tests for the Mutate function"""
    
    def test_mutate_length(self):
        """Mutated sequence should have same length"""
        seq = np.array([1, 1, -1, 1, -1])
        mutated = Mutate(seq, 0.5)
        self.assertEqual(len(mutated), len(seq))
    
    def test_mutate_does_not_modify_original(self):
        """Mutate should not modify the original sequence"""
        seq = np.array([1, 1, -1, 1, -1])
        original = seq.copy()
        _ = Mutate(seq, 0.5)
        np.testing.assert_array_equal(seq, original)
    
    def test_mutate_zero_probability(self):
        """Zero mutation probability should not change sequence"""
        seq = np.array([1, 1, -1, 1, -1])
        mutated = Mutate(seq, 0.0)
        np.testing.assert_array_equal(seq, mutated)
    
    def test_mutate_values(self):
        """Mutated sequence should only contain -1 and 1"""
        seq = np.array([1, 1, -1, 1, -1, 1, -1, 1, -1, 1])
        for _ in range(50):
            mutated = Mutate(seq, 0.5)
            for val in mutated:
                self.assertIn(val, [-1, 1])


class TestBitstringToSpin(unittest.TestCase):
    """Tests for bitstring to spin conversion"""
    
    def test_basic_conversion(self):
        """Test basic bitstring conversion"""
        result = bitstring_to_spin("101", 3)
        expected = np.array([1, -1, 1], dtype=np.int8)
        np.testing.assert_array_equal(result, expected)
    
    def test_all_zeros(self):
        """All zeros should become all -1"""
        result = bitstring_to_spin("0000", 4)
        expected = np.array([-1, -1, -1, -1], dtype=np.int8)
        np.testing.assert_array_equal(result, expected)
    
    def test_all_ones(self):
        """All ones should become all +1"""
        result = bitstring_to_spin("1111", 4)
        expected = np.array([1, 1, 1, 1], dtype=np.int8)
        np.testing.assert_array_equal(result, expected)
    
    def test_padding(self):
        """Short bitstrings should be zero-padded"""
        result = bitstring_to_spin("1", 4)
        expected = np.array([-1, -1, -1, 1], dtype=np.int8)
        np.testing.assert_array_equal(result, expected)
    
    def test_space_removal(self):
        """Spaces in bitstring should be removed"""
        result = bitstring_to_spin("1 0 1", 3)
        expected = np.array([1, -1, 1], dtype=np.int8)
        np.testing.assert_array_equal(result, expected)


class TestGetInteractions(unittest.TestCase):
    """Tests for interaction generation"""
    
    def test_g2_not_empty_for_n_ge_3(self):
        """G2 should not be empty for N >= 3"""
        G2, _ = get_interactions(5)
        self.assertGreater(len(G2), 0)
    
    def test_g4_not_empty_for_n_ge_4(self):
        """G4 should not be empty for N >= 4"""
        _, G4 = get_interactions(6)
        self.assertGreater(len(G4), 0)
    
    def test_g2_indices_valid(self):
        """G2 indices should be within range [0, N-1]"""
        N = 10
        G2, _ = get_interactions(N)
        for pair in G2:
            self.assertGreaterEqual(pair[0], 0)
            self.assertLess(pair[0], N)
            self.assertGreaterEqual(pair[1], 0)
            self.assertLess(pair[1], N)
            self.assertNotEqual(pair[0], pair[1])
    
    def test_g4_indices_valid(self):
        """G4 indices should be within range [0, N-1] and all different"""
        N = 10
        _, G4 = get_interactions(N)
        for quad in G4:
            self.assertEqual(len(set(quad)), 4)  # All different
            for idx in quad:
                self.assertGreaterEqual(idx, 0)
                self.assertLess(idx, N)
    
    def test_flatten_g2(self):
        """Test G2 flattening"""
        G2 = [[0, 1], [1, 2], [2, 3]]
        G2_i, G2_j = flatten_G2(G2)
        self.assertEqual(G2_i, [0, 1, 2])
        self.assertEqual(G2_j, [1, 2, 3])
    
    def test_flatten_g4(self):
        """Test G4 flattening"""
        G4 = [[0, 1, 2, 3], [1, 2, 3, 4]]
        a, b, c, d = flatten_G4(G4)
        self.assertEqual(a, [0, 1])
        self.assertEqual(b, [1, 2])
        self.assertEqual(c, [2, 3])
        self.assertEqual(d, [3, 4])


class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def test_energy_after_flip(self):
        np.random.seed(123)
        seq = np.random.choice([-1, 1], size=20)
        original_energy = energy(seq)
        
        energy_changed = False
        for i in range(len(seq)):
            flipped = flip(seq, i)
            if energy(flipped) != original_energy:
                energy_changed = True
                break
        
        self.assertTrue(energy_changed)
    
    def test_combine_mutate_preserves_validity(self):
        np.random.seed(456)
        p1 = np.random.choice([-1, 1], size=30)
        p2 = np.random.choice([-1, 1], size=30)
        
        child = Combine(p1, p2)
        mutated = Mutate(child, 0.1)
        
        self.assertEqual(len(mutated), 30)
        
        for val in mutated:
            self.assertIn(val, [-1, 1])
        
        e = energy(mutated)
        self.assertGreaterEqual(e, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
