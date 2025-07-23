import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
import os


class SudokuDataset(Dataset):
    """Dataset for Sudoku puzzles with configurable representations."""
    
    def __init__(
        self,
        csv_path: str,
        split: str = 'train',
        max_samples: Optional[int] = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        representation: str = 'one_hot',  # 'one_hot' or 'embedding'
        seed: int = 42
    ):
        """
        Initialize Sudoku dataset.
        
        Args:
            csv_path: Path to CSV file with puzzles
            split: 'train', 'val', or 'test'
            max_samples: Limit number of samples loaded
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            representation: How to encode inputs ('one_hot' or 'embedding')
            seed: Random seed for splits
        """
        self.representation = representation
        self.split = split
        
        # Load data
        df = pd.read_csv(csv_path, nrows=max_samples)
        
        # Split data
        np.random.seed(seed)
        n_samples = len(df)
        indices = np.random.permutation(n_samples)
        
        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)
        
        if split == 'train':
            indices = indices[:train_end]
        elif split == 'val':
            indices = indices[train_end:val_end]
        else:  # test
            indices = indices[val_end:]
        
        self.puzzles = df.iloc[indices]['quizzes'].values
        self.solutions = df.iloc[indices]['solutions'].values
        
        # Calculate difficulty statistics
        self.difficulties = [self._count_empty_cells(p) for p in self.puzzles]
        
    def __len__(self) -> int:
        return len(self.puzzles)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Get a puzzle-solution pair.
        
        Returns:
            puzzle: Input tensor representation
            solution: Target tensor (indices 0-8 for digits 1-9)
            info: Dictionary with metadata
        """
        puzzle_str = self.puzzles[idx]
        solution_str = self.solutions[idx]
        
        # Parse strings to numpy arrays
        puzzle = np.array([int(c) for c in puzzle_str]).reshape(9, 9)
        solution = np.array([int(c) for c in solution_str]).reshape(9, 9)
        
        # Create input representation
        if self.representation == 'one_hot':
            # One-hot encoding: 81 positions × 10 dimensions (0-9)
            puzzle_tensor = self._to_one_hot(puzzle)
        else:
            # Direct encoding for embedding approach
            puzzle_tensor = torch.tensor(puzzle, dtype=torch.long).flatten()
        
        # Target: convert 1-9 to 0-8 for classification
        solution_tensor = torch.tensor(solution - 1, dtype=torch.long).flatten()
        
        # Metadata
        info = {
            'difficulty': self.difficulties[idx],
            'puzzle_str': puzzle_str,
            'solution_str': solution_str,
            'empty_cells': (puzzle == 0).sum().item()
        }
        
        return puzzle_tensor, solution_tensor, info
    
    def _to_one_hot(self, puzzle: np.ndarray) -> torch.Tensor:
        """Convert puzzle to one-hot encoding."""
        # Create one-hot tensor: 81 positions × 10 classes (0-9)
        one_hot = torch.zeros(81, 10)
        flat_puzzle = puzzle.flatten()
        
        for i, val in enumerate(flat_puzzle):
            one_hot[i, val] = 1.0
        
        # Flatten to single vector
        return one_hot.flatten()  # Shape: (810,)
    
    def _count_empty_cells(self, puzzle_str: str) -> int:
        """Count number of empty cells as simple difficulty measure."""
        return puzzle_str.count('0')
    
    @staticmethod
    def decode_puzzle(tensor: torch.Tensor, representation: str = 'one_hot') -> np.ndarray:
        """Decode tensor back to 9x9 puzzle array."""
        if representation == 'one_hot':
            # Reshape to (81, 10) and take argmax
            reshaped = tensor.reshape(81, 10)
            values = reshaped.argmax(dim=1)
        else:
            values = tensor
        
        return values.cpu().numpy().reshape(9, 9)
    
    @staticmethod
    def decode_solution(tensor: torch.Tensor) -> np.ndarray:
        """Decode solution tensor back to 9x9 array with digits 1-9."""
        # Add 1 to convert from 0-8 back to 1-9
        values = tensor.cpu().numpy() + 1
        return values.reshape(9, 9)
    
    def get_difficulty_stats(self) -> dict:
        """Get statistics about puzzle difficulties in this split."""
        difficulties = np.array(self.difficulties)
        return {
            'mean_empty': difficulties.mean(),
            'std_empty': difficulties.std(),
            'min_empty': difficulties.min(),
            'max_empty': difficulties.max(),
            'histogram': np.histogram(difficulties, bins=10)[0].tolist()
        }


class SudokuConstraintChecker:
    """Utility class for checking Sudoku constraints."""
    
    @staticmethod
    def is_valid_sudoku(grid: np.ndarray) -> bool:
        """Check if a completed Sudoku grid is valid."""
        # Check rows
        for row in grid:
            if len(set(row)) != 9 or min(row) < 1 or max(row) > 9:
                return False
        
        # Check columns
        for col in grid.T:
            if len(set(col)) != 9:
                return False
        
        # Check 3x3 boxes
        for box_row in range(3):
            for box_col in range(3):
                box = grid[box_row*3:(box_row+1)*3, box_col*3:(box_col+1)*3]
                if len(set(box.flatten())) != 9:
                    return False
        
        return True
    
    @staticmethod
    def count_violations(grid: np.ndarray) -> int:
        """Count total constraint violations in a grid."""
        violations = 0
        
        # Check rows
        for row in grid:
            non_zero = row[row > 0]
            violations += len(non_zero) - len(set(non_zero))
        
        # Check columns
        for col in grid.T:
            non_zero = col[col > 0]
            violations += len(non_zero) - len(set(non_zero))
        
        # Check boxes
        for box_row in range(3):
            for box_col in range(3):
                box = grid[box_row*3:(box_row+1)*3, box_col*3:(box_col+1)*3]
                non_zero = box.flatten()
                non_zero = non_zero[non_zero > 0]
                violations += len(non_zero) - len(set(non_zero))
        
        return violations


def create_sudoku_dataloaders(
    csv_path: str,
    batch_size: int = 32,
    max_samples: Optional[int] = None,
    representation: str = 'one_hot',
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders."""
    
    train_dataset = SudokuDataset(csv_path, 'train', max_samples, representation=representation)
    val_dataset = SudokuDataset(csv_path, 'val', max_samples, representation=representation)
    test_dataset = SudokuDataset(csv_path, 'test', max_samples, representation=representation)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataset
    csv_path = os.path.join(os.path.dirname(__file__), "sudoku.csv")
    
    # Create dataset with small sample
    dataset = SudokuDataset(csv_path, max_samples=100, representation='one_hot')
    print(f"Dataset size: {len(dataset)}")
    print(f"Difficulty stats: {dataset.get_difficulty_stats()}")
    
    # Test single sample
    puzzle, solution, info = dataset[0]
    print(f"\nSample puzzle shape: {puzzle.shape}")
    print(f"Sample solution shape: {solution.shape}")
    print(f"Sample info: {info}")
    
    # Decode and verify
    decoded_puzzle = SudokuDataset.decode_puzzle(puzzle, 'one_hot')
    decoded_solution = SudokuDataset.decode_solution(solution)
    
    print(f"\nDecoded puzzle:\n{decoded_puzzle}")
    print(f"\nDecoded solution:\n{decoded_solution}")
    print(f"Solution valid: {SudokuConstraintChecker.is_valid_sudoku(decoded_solution)}")