
import unittest
from metient.util.vertex_labeling_util import pareto_front

class TestAncestralLabelingMetrics(unittest.TestCase):

    def test_clear_dominance(self):
        
        # Test 1: Simple case - one dominates all others
        print("Test 1: Clear dominance")
        all_pars_metrics = [
            (5, 5, 5),
            (3, 3, 3),  # Dominates all others
            (4, 4, 4),
            (6, 6, 6),
        ]
        solutions = ['A', 'B', 'C', 'D']
        metrics, sols = pareto_front(solutions, all_pars_metrics)
        print(f"Metrics: {metrics}")
        print(f"Solutions: {sols}")
        assert metrics == [(3, 3, 3)], f"Expected [(3, 3, 3)], got {metrics}"
        print("✓ Passed\n")
    
    def test_no_dominance(self):

        # Test 2: No dominance - all are Pareto optimal
        print("Test 2: No dominance - all Pareto optimal")
        all_pars_metrics = [
            (1, 5, 9),  # Best in obj 1
            (5, 1, 9),  # Best in obj 2
            (5, 5, 1),  # Best in obj 3
        ]
        solutions = ['A', 'B', 'C']
        metrics, sols = pareto_front(solutions, all_pars_metrics)
        print(f"Metrics: {metrics}")
        print(f"Solutions: {sols}")
        assert len(metrics) == 3, f"Expected 3 solutions, got {len(metrics)}"
        print("✓ Passed\n")
    
    def test_mixed_dominance(self):

        # Test 3: Mixed - some dominated, some not
        print("Test 3: Mixed dominance")
        all_pars_metrics = [
            (2, 2, 5),  # A - Pareto optimal
            (5, 5, 2),  # B - Pareto optimal  
            (3, 3, 6),  # C - DOMINATED by A (2<3, 2<3, 5<6,)
            (1, 6, 6),  # D - Pareto optimal (best in obj 1)
            (6, 1, 6),  # E - Pareto optimal (best in obj 2)
        ]
        solutions = ['A', 'B', 'C', 'D', 'E']
        metrics, sols = pareto_front(solutions, all_pars_metrics)
        print(f"Metrics: {metrics}")
        print(f"Solutions: {sols}")
        assert len(metrics) == 4, f"Expected 4 solutions, got {len(metrics)}"
        assert (3, 3, 6) not in metrics, "Solution C should be dominated by A"
    
    def test_chain_dominance(self):

        # Test 4: Chain dominance
        print("Test 4: Chain dominance")
        all_pars_metrics = [
            (1, 1, 1),  # Dominates all
            (2, 2, 2),  # Dominated by first
            (3, 3, 3),  # Dominated by first and second
            (4, 4, 4),  # Dominated by all above
        ]
        solutions = ['A', 'B', 'C', 'D']
        metrics, sols = pareto_front(solutions, all_pars_metrics)
        print(f"Metrics: {metrics}")
        print(f"Solutions: {sols}")
        assert metrics == [(1, 1, 1)], f"Expected only (1,1,1), got {metrics}"
        print("✓ Passed\n")
    
    def test_identical_solutions(self):

        # Test 5: Identical solutions
        print("Test 5: Identical metrics")
        all_pars_metrics = [
            (3, 3, 3),
            (3, 3, 3),  # Duplicate
            (3, 3, 3),  # Duplicate
        ]
        solutions = ['A', 'B', 'C']
        metrics, sols = pareto_front(solutions, all_pars_metrics)
        print(f"Metrics: {metrics}")
        print(f"Solutions: {sols}")
        # All should be kept (none strictly dominates the others)
        assert len(metrics) == 3, f"Expected 3 solutions, got {len(metrics)}"
        print("✓ Passed\n")
    
    def test_partial_dominance(self):

        # Test 6: Partial dominance
        print("Test 6: Partial dominance")
        all_pars_metrics = [
            (1, 5, 5),  # Pareto optimal
            (2, 4, 5),  # Pareto optimal (better in obj 2 than first)
            (5, 5, 1),  # Pareto optimal
        ]
        solutions = ['A', 'B', 'C']
        metrics, sols = pareto_front(solutions, all_pars_metrics)
        print(f"Metrics: {metrics}")
        print(f"Solutions: {sols}")
        # A doesn't dominate B (5≤4 fails in obj 2)
        # So all 3 should be Pareto optimal
        assert len(metrics) == 3, f"Expected 3 solutions, got {len(metrics)}"
        print("✓ Passed\n")
    
    def test_single_solution(self):

        # Test 7: Edge case - single solution
        print("Test 7: Single solution")
        all_pars_metrics = [(5, 5, 5)]
        solutions = ['A']
        metrics, sols = pareto_front(solutions, all_pars_metrics)
        print(f"Metrics: {metrics}")
        print(f"Solutions: {sols}")
        assert metrics == [(5, 5, 5)], f"Expected [(5,5,5)], got {metrics}"
        print("✓ Passed\n")
    
    def test_dominance(self):

        # Test 8: Real dominance scenario
        print("Test 8: Clear dominated solution")
        all_pars_metrics = [
            (2, 3, 4),  # Pareto optimal
            (3, 4, 5),  # DOMINATED by first (2≤3, 3≤4, 4≤5, and at least one strict)
            (1, 5, 5),  # Pareto optimal (better in obj 1)
        ]
        solutions = ['A', 'B', 'C']
        metrics, sols = pareto_front(solutions, all_pars_metrics)
        print(f"Metrics: {metrics}")
        print(f"Solutions: {sols}")
        assert (3, 4, 5) not in metrics, "Solution B should be dominated by A"
        assert len(metrics) == 2, f"Expected 2 solutions, got {len(metrics)}"
        print("✓ Passed\n")
    
    def test_two_obj(self):

        # Test 9: Two objectives instead of three
        print("Test 9: Two objectives")
        all_pars_metrics = [
            (1, 3),  # A - Pareto optimal
            (3, 1),  # B - Pareto optimal
            (2, 2),  # C - Pareto optimal (neither A nor B dominates it)
            (2, 4),  # D - DOMINATED by A (1<2, 3<4)
            (4, 2),  # E - DOMINATED by B (3<4, 1<2)
        ]
        solutions = ['A', 'B', 'C', 'D', 'E']
        metrics, sols = pareto_front(solutions, all_pars_metrics)
        print(f"Metrics: {metrics}")
        print(f"Solutions: {sols}")
        assert len(metrics) == 3, f"Expected 3 solutions, got {len(metrics)}"
        assert (2, 4) not in metrics and (4, 2) not in metrics, "D and E should be dominated"
    
if __name__ == "__main__":
    unittest.main()