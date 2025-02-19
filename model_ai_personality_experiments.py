#!/usr/bin/env python
# coding: utf-8

# In[28]:


#!/usr/bin/env python
# coding: utf-8

# %% [markdown]
# # Wordle Solver Optimization Experiment
# Comparing different programming personalities' approaches to optimizing a Wordle solver

# %% Imports and Setup
import boto3
import inspect
import re
import json
from datetime import datetime
from dataclasses import dataclass
import pandas as pd
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import time
import random
from wordle_simulator import WordleSimulator


# In[42]:


@dataclass
class SolverPerformance:
    personality_name: str
    win_rate: float
    avg_attempts: float
    avg_guess_time: float


# In[62]:


# Create experiment directory
EXPERIMENT_DIR = Path("experiments")
EXPERIMENT_DIR.mkdir(exist_ok=True)

# %% Configuration
NUM_TEST_GAMES = 500
MAX_ATTEMPTS = 6


# In[4]:


# %% Load Word Lists
def load_word_list(filename: str) -> List[str]:
    """Load and clean word list from file"""
    with open(filename, 'r') as f:
        return [word.strip().lower() for word in f if len(word.strip()) == 5]
# Load word lists
irish_words = load_word_list('./simple_wordle/assets/irish_words.txt')
english_words = load_word_list('./simple_wordle/assets/english_words.txt')


# In[5]:


# %% Random Guessing Baseline Test
# Create a very simple solver that just guesses randomly
random_solver_code = """
import random

class WordleSolver:
    def __init__(self, word_list):
        self.all_words = word_list[:]
        self.possible_words = word_list[:]
        self.guesses = []
        print(f"Initialized with {len(self.all_words)} words")
    
    def make_guess(self):
        \"\"\"Simply returns a random word from the full word list\"\"\"
        if not self.all_words:
            return None
        guess = random.choice(self.all_words)
        self.guesses.append(guess)
        return guess
    
    def evaluate_guess(self, guess: str, secret_word: str) -> list:
        \"\"\"
        Evaluate a guess against the secret word.
        Returns a list of clues: 'Green', 'Yellow', or 'Gray' for each letter.
        \"\"\"
        if len(guess) != len(secret_word):
            raise ValueError(f"Words must be same length: guess={len(guess)}, secret={len(secret_word)}")
            
        clues = ['Gray'] * len(guess)
        secret_letters = list(secret_word)
        
        # First pass: mark Greens (exact matches)
        for i, letter in enumerate(guess):
            if secret_word[i] == letter:
                clues[i] = 'Green'
                secret_letters[i] = None
        
        # Second pass: mark Yellows (right letter, wrong position)
        for i, letter in enumerate(guess):
            if clues[i] == 'Gray' and letter in secret_letters:
                clues[i] = 'Yellow'
                secret_letters[secret_letters.index(letter)] = None
                
        return clues
    
    def submit_guess_and_get_clues(self, clues, guess):
        \"\"\"Random solver doesn't use clues, so this does nothing\"\"\"
        pass
    
    def reset(self):
        \"\"\"Reset the solver state\"\"\"
        self.possible_words = self.all_words[:]
        self.guesses = []
"""

print("Creating test module...")
# Create a module for our random solver
import types
mod = types.ModuleType('random_solver')
exec(random_solver_code, mod.__dict__)
solver_class = mod.WordleSolver

# Run simulation with first 100 words
test_words = irish_words[:100]
print(f"Testing with {len(test_words)} words...")

win_rate, avg_attempts, avg_guess_time = WordleSimulator.simulate_games(
    solver_class,
    test_words,
    num_games=NUM_TEST_GAMES,
    max_attempts=MAX_ATTEMPTS
)

print("\nRandom Guessing Baseline Results:")
print(f"Win Rate: {win_rate * 100:.1f}%")
print(f"Avg Attempts for Wins: {avg_attempts:.2f}")
print(f"Avg Guess Time: {avg_guess_time * 1000:.3f}ms")


# In[63]:


# %% Generate Fixed Word List for Testing

# Choose the language for testing ('english' or 'irish')
language = 'english'  # Change this to 'irish' to test with the Irish word list
word_list = english_words if language == 'english' else irish_words

# Generate a fixed list of secret words
random.seed(1337)  # Use a fixed seed for reproducibility
fixed_secret_words = random.sample(word_list, NUM_TEST_GAMES)
print(f"Generated fixed list of {len(fixed_secret_words)} secret words.")
#print(fixed_secret_words) # Uncomment to see the words


# # Prompt
# 
# ## Famous Computer Scientists
# 
# 1. You are John Carmack, a legendary game programmer known for your focus on optimization, performance, and low-level techniques. Your task is to improve a basic Python Wordle solver (provided below) to make it as good as possible.
# 2. You are Donald Knuth, a renowned computer scientist known for your emphasis on algorithmic elegance, correctness, and thorough documentation. Your task is to improve a basic Python Wordle solver (provided below), focusing on creating a well-documented, provably correct, and efficient algorithm. Your task is to build an implementation for a basic Python Wordle solver (details provided below) to make it as good as possible.
# 3. You are Martin Fowler, a software engineer known for your expertise in clean code, design patterns, and refactoring. Your task is to improve a basic Python Wordle solver (provided below), focusing on readability, maintainability, and the application of appropriate design principles. Your task is to build an implementation for a basic Python Wordle solver (details provided below) to make it as good as possible.
# 4. You are Linus Torvalds, a expert software engineer known for creating Linux and Git, and for your focus on pragmatic efficiency, performance, and robustness. Your task is to improve a basic Python Wordle solver (provided below).  Your task is to build an implementation for a basic Python Wordle solver (details provided below) to make it as good as possible.
# 5. You are Guido van Rossum, creator of Python, known for emphasizing code readability and the 'one obvious way to do it' philosophy. You believe in clean, explicit code that follows the principle of least surprise. Your task is to build an implementation for a basic Python Wordle solver (details provided below) to make it as good as possible.
# 6. You are Katherine Johnson, a NASA mathematician known for your extraordinary precision and verification of computer calculations for critical space missions. You believe in thorough checking, mathematical rigor, and the importance of understanding why each calculation works, not just that it works. Your task is to build an implementation for a basic Python Wordle solver (details provided below) to make it as good as possible.
# 7. You are Travis Oliphant, creator of NumPy and SciPy, known for building foundational tools that transformed Python into a powerful scientific computing platform. You excel at optimizing numerical computations and creating clean APIs that make complex operations accessible. Your task is to build an implementation for a basic Python Wordle solver (details provided below) to make it as good as possible.
# 8. You are Stanley Seibert, expert of HPC at Anaconda, known for your expertise in high-performance computing, the numba library. You focus on maximizing computational efficiency while maintaining code readability and maintainability. Your task is to build an implementation for a basic Python Wordle solver (details provided below) to make it as good as possible.
# 9. Hybrid - You are an expert genius engineer a superposition of the very best traits and knowledge of John Carmack, Donald Knuth, Katherine Johnson. Your task is to improve a basic Python Wordle solver (provided below) to make it as good as possible. 
# 
# 
# **ABSOLUTELY CRITICAL REQUIREMENTS (MUST READ CAREFULLY):**
# 
# 1.  **Class Name:** The class name *MUST* be `WordleSolver`.
# 
# 2.  **Required Interface:** The class *MUST* implement the following four methods with *EXACTLY* these signatures (including type hints). Copy and paste these directly into your code:
# 
#     ```python
#     class WordleSolver:
#         def __init__(self, word_list: list[str]):
#             """
#             Initializes the solver with a list of possible words.
#             """
#             pass
# 
#         def make_guess(self) -> str:
#             """
#             Returns the solver's next guess (a 5-letter strin that must be a possible word in the list).
#             Uses the internal state (e.g., possible_words) to make a guess.
#             Returns None if no guess is possible.
#             """
#             pass
# 
#         def submit_guess_and_get_clues(self, clues: list[str], guess: str):
#             """
#             Updates the solver's internal state based on the provided clues from a guess.
#         
#             Args:
#                 clues: A list of strings representing the feedback for each letter in the guess.
#                        Each element corresponds to a letter in the guess and can be:
#                          - "Green":  The letter is in the correct position.
#                          - "Yellow": The letter is in the word but in a different position.
#                          - "Gray":   The letter is not in the word.
#         
#                        Note on repeated letters:
#                        - If a guessed word contains multiple instances of a letter, the feedback
#                          ("Green", "Yellow", "Gray") for each instance is determined based on the
#                          actual occurrences in the secret word.
#                        - For example, if the secret word is "BALLOON" and the guess is "LLAMA":
#                          - The first 'L' in the guess may receive a "Yellow" if 'L' is in the word
#                            but not in that position.
#                          - The second 'L' in the guess may receive a "Green" if it matches the 'L'
#                            in the secret word at that position.
#                          - Additional 'L's in the guess beyond the count in the secret word will
#                            receive "Gray".
#         
#                 guess: The word that was guessed (a 5-letter string).
#         
#             Updates the solver's possible word list by eliminating words that are inconsistent
#             with the provided clues.
#             """
# 
#             pass
# 
#         def reset(self):
#             """
#             Resets the solver's internal state to start a new game.
#             """
#             pass
#     ```
# 
#     *   **Do NOT add any other public methods.**
#     *   **Do NOT change the method names, parameter names, or types.**
#     *   You *may* add private helper methods (starting with an underscore, e.g., `_my_helper_method`), but *ONLY* if necessary.
# 
# 3.  **NO `evaluate_guess` METHOD:** Do *NOT*, under *any* circumstances, include a method named `evaluate_guess` within your `WordleSolver` class. The clue generation is handled *externally*. **Including this method will result in automatic rejection.**
# 
# 4.  **SINGLE CODE BLOCK OUTPUT:** Your *entire* response *MUST* be a *SINGLE*, properly formatted Markdown code block. Start with three backticks, the word `python`, a newline, your code, a newline, and three closing backticks. Like this:
# 
#     ```python
#     class WordleSolver:
#         # ... your implementation ...
#     ```
# 
#     *   **Do NOT include any text or commentary outside of this code block.** Your response should begin and end with the triple backticks.
#     *   **Do NOT include any testing code.**
#     *   **Do NOT include any import statements unless absolutely necessary.**
#     *   **WARNING:** Any response that includes text outside the code block will be automatically rejected.  Be extremely careful to follow this rule.
# 
# 5.  **Error Handling:**
#     *   The `word_list` provided in `__init__` may be empty or very short, or contain a up to 15k words. Handle these cases without crashing.
#     *   Do not make assumptions about input validation beyond the type hints.
# 
# 6.  **Performance:** Aim for maximum efficiency, but *NEVER* at the cost of violating the above requirements.

# In[35]:


class WordleSolverNoPers1:
    def __init__(self, word_list: list[str]):
        """
        Initializes the solver with a list of possible words.
        """
        self.original_words = [word.upper() for word in word_list]
        self.possible_words = self.original_words.copy()
        self.letter_freqs = self._calculate_letter_frequencies()
        
    def make_guess(self) -> str:
        """
        Returns the solver's next guess (a 5-letter string that must be a possible word in the list).
        Uses the internal state (e.g., possible_words) to make a guess.
        Returns None if no guess is possible.
        """
        if not self.possible_words:
            return None
            
        if len(self.possible_words) == 1:
            return self.possible_words[0]
            
        # Score words based on letter frequency and position
        best_score = -1
        best_word = self.possible_words[0]
        
        for word in self.possible_words:
            score = self._score_word(word)
            if score > best_score:
                best_score = score
                best_word = word
                
        return best_word

    def submit_guess_and_get_clues(self, clues: list[str], guess: str):
        """
        Updates the solver's internal state based on the provided clues from a guess.
        """
        guess = guess.upper()
        new_possible = []
        
        for word in self.possible_words:
            if self._is_word_possible(word, guess, clues):
                new_possible.append(word)
                
        self.possible_words = new_possible
        self.letter_freqs = self._calculate_letter_frequencies()

    def reset(self):
        """
        Resets the solver's internal state to start a new game.
        """
        self.possible_words = self.original_words.copy()
        self.letter_freqs = self._calculate_letter_frequencies()

    def _calculate_letter_frequencies(self) -> dict:
        """
        Calculate letter frequencies for current possible words
        """
        freqs = {}
        total_words = len(self.possible_words)
        if total_words == 0:
            return freqs
            
        for pos in range(5):
            freqs[pos] = {}
            for word in self.possible_words:
                letter = word[pos]
                freqs[pos][letter] = freqs[pos].get(letter, 0) + 1
                
        return freqs

    def _score_word(self, word: str) -> float:
        """
        Score a word based on letter frequencies and positions
        """
        if not self.letter_freqs:
            return 0
            
        score = 0
        seen = set()
        
        for pos, letter in enumerate(word):
            if letter not in seen:
                score += self.letter_freqs[pos].get(letter, 0)
                seen.add(letter)
                
        return score / len(self.possible_words)

    def _is_word_possible(self, word: str, guess: str, clues: list[str]) -> bool:
        """
        Check if a word is still possible given the guess and clues
        """
        word = word.upper()
        remaining_letters = list(word)
        
        # First check green clues
        for i, (letter, clue) in enumerate(zip(guess, clues)):
            if clue == "Green" and word[i] != letter:
                return False
            if clue == "Green" and letter in remaining_letters:
                remaining_letters.remove(letter)
                
        # Then check yellow clues
        for i, (letter, clue) in enumerate(zip(guess, clues)):
            if clue == "Yellow":
                if word[i] == letter:  # Letter can't be in this position
                    return False
                if letter not in remaining_letters:  # Letter must be somewhere
                    return False
                if letter in remaining_letters:
                    remaining_letters.remove(letter)
                    
        # Finally check gray clues
        for i, (letter, clue) in enumerate(zip(guess, clues)):
            if clue == "Gray" and letter in remaining_letters:
                return False
                
        return True


# In[36]:


class WordleSolverNoPers2:
    def __init__(self, word_list: list[str]):
        """
        Initializes the solver with a list of possible words.
        """
        self.original_words = word_list.copy()
        self.possible_words = word_list.copy()
        self.last_guess = None

    def make_guess(self) -> str:
        """
        Returns the solver's next guess (a 5-letter string that must be a possible word in the list).
        Uses the internal state (e.g., possible_words) to make a guess.
        Returns None if no guess is possible.
        """
        if not self.possible_words:
            return None
            
        # If this is first guess, try to pick a word with common letters
        if len(self.possible_words) == len(self.original_words):
            letter_freq = {}
            for word in self.possible_words:
                for letter in set(word):
                    letter_freq[letter] = letter_freq.get(letter, 0) + 1
                    
            word_scores = {}
            for word in self.possible_words:
                score = sum(letter_freq[letter] for letter in set(word))
                word_scores[word] = score
                
            best_word = max(word_scores.items(), key=lambda x: x[1])[0]
            self.last_guess = best_word
            return best_word
            
        # Otherwise pick first remaining possible word
        self.last_guess = self.possible_words[0]
        return self.last_guess

    def submit_guess_and_get_clues(self, clues: list[str], guess: str):
        """
        Updates the solver's internal state based on the provided clues from a guess.
        """
        if not clues or not guess:
            return
            
        new_possible = []
        
        for possible_word in self.possible_words:
            if self._is_word_possible(possible_word, guess, clues):
                new_possible.append(possible_word)
                
        self.possible_words = new_possible

    def reset(self):
        """
        Resets the solver's internal state to start a new game.
        """
        self.possible_words = self.original_words.copy()
        self.last_guess = None
        
    def _is_word_possible(self, word: str, guess: str, clues: list[str]) -> bool:
        """
        Helper method to check if a word is still possible given the clues
        """
        if word == guess:
            return all(clue == "Green" for clue in clues)
            
        # Check green clues first
        for i, (clue, guess_letter) in enumerate(zip(clues, guess)):
            if clue == "Green" and word[i] != guess_letter:
                return False
                
        # Count remaining letters for yellow/gray checks
        word_letters = list(word)
        guess_letters = list(guess)
        
        # Remove green matches
        for i, (clue, guess_letter) in enumerate(zip(clues, guess)):
            if clue == "Green":
                word_letters[i] = None
                guess_letters[i] = None
                
        # Check yellow clues
        for i, (clue, guess_letter) in enumerate(zip(clues, guess_letters)):
            if clue == "Yellow":
                if guess_letter is None:
                    continue
                    
                # Find matching letter in remaining word letters
                found = False
                for j, word_letter in enumerate(word_letters):
                    if word_letter == guess_letter:
                        word_letters[j] = None
                        found = True
                        break
                if not found:
                    return False
                    
        # Check gray clues
        for i, (clue, guess_letter) in enumerate(zip(clues, guess_letters)):
            if clue == "Gray" and guess_letter in word_letters:
                return False
                
        return True


# In[49]:


class WordleSolverNoPers3:
    def __init__(self, word_list: list[str]):
        """
        Initializes the solver with a list of possible words.
        """
        self.original_words = list(word_list)
        self.possible_words = list(word_list)
        self.last_guess = None
        
    def make_guess(self) -> str:
        """
        Returns the solver's next guess (a 5-letter strin that must be a possible word in the list).
        Uses the internal state (e.g., possible_words) to make a guess.
        Returns None if no guess is possible.
        """
        if not self.possible_words:
            return None
            
        if len(self.possible_words) == 1:
            return self.possible_words[0]
            
        # Score words based on letter frequency and position
        word_scores = {}
        letter_freq = {}
        pos_freq = [{} for _ in range(5)]
        
        # Calculate letter frequencies
        for word in self.possible_words:
            for i, letter in enumerate(word):
                letter_freq[letter] = letter_freq.get(letter, 0) + 1
                pos_freq[i][letter] = pos_freq[i].get(letter, 0) + 1
                
        # Score each word
        for word in self.possible_words:
            score = 0
            seen = set()
            for i, letter in enumerate(word):
                if letter not in seen:
                    score += letter_freq[letter]
                    score += pos_freq[i][letter] * 2
                    seen.add(letter)
            word_scores[word] = score
            
        # Return word with highest score
        return max(word_scores, key=word_scores.get)

    def submit_guess_and_get_clues(self, clues: list[str], guess: str):
        """
        Updates the solver's internal state based on the provided clues from a guess.
        """
        self.last_guess = guess
        
        # Create sets of letters based on clues
        green_letters = {(i, guess[i]) for i in range(5) if clues[i] == "Green"}
        yellow_letters = {(i, guess[i]) for i in range(5) if clues[i] == "Yellow"}
        gray_letters = {guess[i] for i in range(5) if clues[i] == "Gray"}
        
        # Count occurrences of each letter in guess
        letter_counts = {}
        for letter in guess:
            letter_counts[letter] = letter_counts.get(letter, 0) + 1
            
        # Filter possible words
        new_possible = []
        for word in self.possible_words:
            valid = True
            
            # Check green letters (must be in correct positions)
            for pos, letter in green_letters:
                if word[pos] != letter:
                    valid = False
                    break
                    
            if not valid:
                continue
                
            # Check yellow letters (must exist but in different positions)
            for pos, letter in yellow_letters:
                if letter not in word or word[pos] == letter:
                    valid = False
                    break
                    
            if not valid:
                continue
                
            # Check gray letters
            word_letter_counts = {}
            for letter in word:
                word_letter_counts[letter] = word_letter_counts.get(letter, 0) + 1
                
            for letter in gray_letters:
                if letter in green_letters or letter in yellow_letters:
                    # If letter appears as green/yellow elsewhere, only check count
                    if word_letter_counts.get(letter, 0) > letter_counts.get(letter, 0):
                        valid = False
                        break
                else:
                    # If fully gray letter, should not appear at all
                    if letter in word:
                        valid = False
                        break
                        
            if valid:
                new_possible.append(word)
                
        self.possible_words = new_possible

    def reset(self):
        """
        Resets the solver's internal state to start a new game.
        """
        self.possible_words = list(self.original_words)
        self.last_guess = None


# In[56]:


class WordleSolverNoPers4:
    def __init__(self, word_list: list[str]):
        """
        Initializes the solver with a list of possible words.
        """
        self.original_words = word_list
        self.possible_words = word_list.copy()
        self.letter_freqs = self._calculate_letter_frequencies()
        self.position_freqs = self._calculate_position_frequencies()
        
    def make_guess(self) -> str:
        """
        Returns the solver's next guess (a 5-letter string that must be a possible word in the list).
        Uses the internal state (e.g., possible_words) to make a guess.
        Returns None if no guess is possible.
        """
        if not self.possible_words:
            return None
            
        if len(self.possible_words) == 1:
            return self.possible_words[0]
            
        # Score each remaining word based on letter frequency and position
        best_score = -1
        best_word = self.possible_words[0]
        
        for word in self.possible_words:
            score = 0
            seen_letters = set()
            
            for i, letter in enumerate(word):
                if letter not in seen_letters:
                    score += self.letter_freqs[letter]
                    score += self.position_freqs[i][letter]
                    seen_letters.add(letter)
                    
            if score > best_score:
                best_score = score
                best_word = word
                
        return best_word

    def submit_guess_and_get_clues(self, clues: list[str], guess: str):
        """
        Updates the solver's internal state based on the provided clues from a guess.
        """
        new_possible = []
        
        for word in self.possible_words:
            if self._is_word_possible(word, guess, clues):
                new_possible.append(word)
                
        self.possible_words = new_possible
        self.letter_freqs = self._calculate_letter_frequencies()
        self.position_freqs = self._calculate_position_frequencies()

    def reset(self):
        """
        Resets the solver's internal state to start a new game.
        """
        self.possible_words = self.original_words.copy()
        self.letter_freqs = self._calculate_letter_frequencies()
        self.position_freqs = self._calculate_position_frequencies()
        
    def _calculate_letter_frequencies(self):
        """Calculate frequency of each letter across all possible words"""
        freqs = {}
        for word in self.possible_words:
            for letter in set(word):  # Count each letter once per word
                freqs[letter] = freqs.get(letter, 0) + 1
        return freqs
        
    def _calculate_position_frequencies(self):
        """Calculate letter frequencies for each position"""
        pos_freqs = [{} for _ in range(5)]
        for word in self.possible_words:
            for i, letter in enumerate(word):
                pos_freqs[i][letter] = pos_freqs[i].get(letter, 0) + 1
        return pos_freqs
        
    def _is_word_possible(self, word: str, guess: str, clues: list[str]) -> bool:
        """Check if a word is possible given the guess and clues"""
        word_letters = list(word)
        guess_letters = list(guess)
        
        # Handle green clues first
        for i, (clue, guess_letter) in enumerate(zip(clues, guess_letters)):
            if clue == "Green":
                if word_letters[i] != guess_letter:
                    return False
                    
        # Count remaining letters after handling greens
        word_letter_counts = {}
        for i, letter in enumerate(word_letters):
            if clues[i] != "Green" or guess_letters[i] != letter:
                word_letter_counts[letter] = word_letter_counts.get(letter, 0) + 1
                
        # Handle yellow and gray clues
        for i, (clue, guess_letter) in enumerate(zip(clues, guess_letters)):
            if clue == "Yellow":
                if guess_letter not in word_letter_counts or word_letter_counts[guess_letter] == 0:
                    return False
                if word_letters[i] == guess_letter:
                    return False
                word_letter_counts[guess_letter] -= 1
            elif clue == "Gray":
                if guess_letter in word_letter_counts and word_letter_counts[guess_letter] > 0:
                    return False
                    
        return True


# In[37]:


class WordleSolverCarmack:
    def __init__(self, word_list: list[str]):
        """
        Initializes the solver with a list of possible words.
        """
        # Store original list for reset and create set of remaining possibilities
        # Use set for O(1) lookups/removal vs O(n) for list
        self._original_words = set(word_list) 
        self.possible_words = self._original_words.copy()
        
        # Pre-compute letter frequencies for faster analysis
        # Store as instance var to avoid recomputing on reset
        self._letter_freqs = {}
        for word in word_list:
            for letter in word:
                self._letter_freqs[letter] = self._letter_freqs.get(letter, 0) + 1
                
    def make_guess(self) -> str:
        """
        Returns the solver's next guess using letter frequency heuristic.
        """
        if not self.possible_words:
            return None
            
        # Score each remaining word based on letter frequencies
        # Use dict comprehension for better performance vs loops
        scores = {word: sum(self._letter_freqs.get(c, 0) for c in set(word)) 
                 for word in self.possible_words}
        
        # Return word with highest score using max() with key function
        return max(scores, key=scores.get)

    def submit_guess_and_get_clues(self, clues: list[str], guess: str):
        """
        Updates possible words based on clues.
        """
        # Create sets for letter counts/positions for O(1) lookups
        green_positions = {i for i, clue in enumerate(clues) if clue == "Green"}
        yellow_positions = {i for i, clue in enumerate(clues) if clue == "Yellow"}
        gray_letters = {guess[i] for i, clue in enumerate(clues) 
                       if clue == "Gray" and guess[i] not in 
                       {guess[j] for j in green_positions | yellow_positions}}
                       
        # Process remaining words using set operations for speed
        self.possible_words = {
            word for word in self.possible_words
            if all(word[i] == guess[i] for i in green_positions) and
               all(word[i] != guess[i] and guess[i] in word 
                   for i in yellow_positions) and
               not any(letter in gray_letters for letter in word)
        }

    def reset(self):
        """
        Resets to initial state using stored original words.
        """
        self.possible_words = self._original_words.copy()


# In[ ]:





# In[38]:


class WordleSolverFowler:
    def __init__(self, word_list: list[str]):
        """
        Initializes the solver with a list of possible words.
        """
        self.original_word_list = word_list
        self.possible_words = word_list.copy()
        self.word_length = 5

    def make_guess(self) -> str:
        """
        Returns the solver's next guess (a 5-letter string that must be a possible word in the list).
        Uses the internal state (e.g., possible_words) to make a guess.
        Returns None if no guess is possible.
        """
        if not self.possible_words:
            return None
        return self._select_best_guess()

    def submit_guess_and_get_clues(self, clues: list[str], guess: str):
        """
        Updates the solver's internal state based on the provided clues from a guess.
        """
        self.possible_words = [word for word in self.possible_words 
                             if self._is_word_consistent(word, guess, clues)]

    def reset(self):
        """
        Resets the solver's internal state to start a new game.
        """
        self.possible_words = self.original_word_list.copy()

    def _select_best_guess(self) -> str:
        """
        Selects the best guess based on letter frequency analysis.
        """
        if len(self.possible_words) == 1:
            return self.possible_words[0]

        letter_scores = self._calculate_letter_scores()
        return max(self.possible_words, 
                  key=lambda word: self._calculate_word_score(word, letter_scores))

    def _calculate_letter_scores(self) -> dict:
        """
        Calculates frequency scores for each letter in each position.
        """
        position_scores = [{} for _ in range(self.word_length)]
        
        for word in self.possible_words:
            for pos, letter in enumerate(word):
                position_scores[pos][letter] = position_scores[pos].get(letter, 0) + 1

        return position_scores

    def _calculate_word_score(self, word: str, letter_scores: list[dict]) -> float:
        """
        Calculates a score for a word based on letter frequencies.
        """
        score = 0
        seen_letters = set()
        
        for pos, letter in enumerate(word):
            if letter not in seen_letters:
                score += letter_scores[pos].get(letter, 0)
                seen_letters.add(letter)
                
        return score

    def _is_word_consistent(self, word: str, guess: str, clues: list[str]) -> bool:
        """
        Checks if a word is consistent with the clues from a previous guess.
        """
        if len(word) != len(guess):
            return False

        word_chars = list(word)
        guess_chars = list(guess)
        
        # Handle green clues first
        for i, (clue, guess_char) in enumerate(zip(clues, guess_chars)):
            if clue == "Green" and word_chars[i] != guess_char:
                return False
            if clue == "Green":
                word_chars[i] = '*'
                guess_chars[i] = '*'

        # Handle yellow and gray clues
        for i, (clue, guess_char) in enumerate(zip(clues, guess_chars)):
            if guess_char == '*':
                continue
                
            if clue == "Yellow":
                if guess_char not in word_chars:
                    return False
                word_chars[word_chars.index(guess_char)] = '*'
            elif clue == "Gray":
                if guess_char in word_chars:
                    return False

        return True


# In[39]:


class WordleSolverKnuth:
    def __init__(self, word_list: list[str]):
        """
        Initializes the solver with a list of possible words.
        
        Args:
            word_list: List of valid 5-letter words to use
            
        Implementation notes:
        - Store original word list for reset capability
        - Initialize possible_words as current working set
        - Handle empty word list case gracefully
        """
        self.original_words = word_list.copy() if word_list else []
        self.possible_words = word_list.copy() if word_list else []
        
        # Letter frequency maps to help make intelligent guesses
        self._letter_freqs = {}
        self._position_freqs = [{} for _ in range(5)]
        self._calculate_frequencies()

    def make_guess(self) -> str:
        """
        Returns the solver's next guess based on current possible words.
        
        Returns:
            str: A 5-letter word guess, or None if no valid guesses remain
            
        Implementation notes:
        - If only one word remains, use it
        - Otherwise score remaining words based on letter frequencies
        - Return highest scoring word as guess
        """
        if not self.possible_words:
            return None
            
        if len(self.possible_words) == 1:
            return self.possible_words[0]
            
        # Score each remaining word based on letter frequencies
        best_score = -1
        best_word = None
        
        for word in self.possible_words:
            # Score based on letter frequency and position frequency
            score = self._score_word(word)
            if score > best_score:
                best_score = score
                best_word = word
                
        return best_word

    def submit_guess_and_get_clues(self, clues: list[str], guess: str):
        """
        Updates possible words based on clue feedback.
        
        Args:
            clues: List of "Green", "Yellow", "Gray" feedback for each letter
            guess: The 5-letter word that was guessed
            
        Implementation notes:
        - Filter possible_words to only keep words matching clue pattern
        - Recalculate letter frequencies after filtering
        """
        new_possible = []
        
        for word in self.possible_words:
            if self._is_word_consistent(word, guess, clues):
                new_possible.append(word)
                
        self.possible_words = new_possible
        self._calculate_frequencies()

    def reset(self):
        """
        Resets solver state to initial conditions.
        
        Implementation notes:
        - Restore original word list
        - Reset letter frequencies
        """
        self.possible_words = self.original_words.copy()
        self._calculate_frequencies()

    def _calculate_frequencies(self):
        """
        Calculate letter and position frequencies for remaining possible words.
        """
        self._letter_freqs.clear()
        for i in range(5):
            self._position_freqs[i].clear()
            
        for word in self.possible_words:
            for i, letter in enumerate(word):
                self._letter_freqs[letter] = self._letter_freqs.get(letter, 0) + 1
                self._position_freqs[i][letter] = self._position_freqs[i].get(letter, 0) + 1

    def _score_word(self, word: str) -> float:
        """
        Score a word based on letter frequencies and uniqueness.
        
        Args:
            word: Word to score
            
        Returns:
            float: Score for the word (higher is better)
        """
        score = 0
        seen = set()
        
        for i, letter in enumerate(word):
            if letter not in seen:
                score += self._letter_freqs.get(letter, 0)
                score += self._position_freqs[i].get(letter, 0) * 0.5
                seen.add(letter)
                
        return score

    def _is_word_consistent(self, word: str, guess: str, clues: list[str]) -> bool:
        """
        Check if a word is consistent with given clues.
        
        Args:
            word: Word to check
            guess: Previous guess
            clues: Clue feedback for guess
            
        Returns:
            bool: True if word is consistent with clues
        """
        # Check green clues first
        for i, (clue, guess_letter) in enumerate(zip(clues, guess)):
            if clue == "Green" and word[i] != guess_letter:
                return False
                
        # Track yellow letters
        remaining_letters = list(word)
        
        # Remove green positions
        for i, (clue, guess_letter) in enumerate(zip(clues, guess)):
            if clue == "Green" and guess_letter in remaining_letters:
                remaining_letters.remove(guess_letter)
                
        # Check yellow clues
        for i, (clue, guess_letter) in enumerate(zip(clues, guess)):
            if clue == "Yellow":
                if word[i] == guess_letter:  # Yellow letter can't be in same position
                    return False
                if guess_letter not in remaining_letters:  # Yellow letter must be somewhere
                    return False
                remaining_letters.remove(guess_letter)
                    
        # Check gray clues
        for i, (clue, guess_letter) in enumerate(zip(clues, guess)):
            if clue == "Gray" and guess_letter in remaining_letters:
                return False
                
        return True


# In[40]:


class WordleSolverTorvalds:
    def __init__(self, word_list: list[str]):
        """
        Initializes the solver with a list of possible words.
        """
        self.original_words = [w.upper() for w in word_list]
        self.possible_words = self.original_words.copy()
        
    def make_guess(self) -> str:
        """
        Returns the solver's next guess (a 5-letter string that must be a possible word in the list).
        Uses the internal state (e.g., possible_words) to make a guess.
        Returns None if no guess is possible.
        """
        if not self.possible_words:
            return None
            
        # Simple frequency analysis for remaining words
        if len(self.possible_words) > 1:
            letter_freq = {}
            for word in self.possible_words:
                for pos, letter in enumerate(word):
                    if (pos, letter) not in letter_freq:
                        letter_freq[(pos, letter)] = 0
                    letter_freq[(pos, letter)] += 1
                        
            # Score words based on unique letter positions
            best_score = -1
            best_word = self.possible_words[0]
            
            for word in self.possible_words:
                score = sum(letter_freq.get((pos, letter), 0) 
                          for pos, letter in enumerate(word))
                if score > best_score:
                    best_score = score
                    best_word = word
            return best_word
            
        return self.possible_words[0]

    def submit_guess_and_get_clues(self, clues: list[str], guess: str):
        """
        Updates the solver's internal state based on the provided clues from a guess.
        """
        guess = guess.upper()
        new_possible = []
        
        for word in self.possible_words:
            if self._matches_clues(word, guess, clues):
                new_possible.append(word)
                
        self.possible_words = new_possible

    def reset(self):
        """
        Resets the solver's internal state to start a new game.
        """
        self.possible_words = self.original_words.copy()
        
    def _matches_clues(self, word: str, guess: str, clues: list[str]) -> bool:
        """
        Helper method to check if a word matches the clues from a guess
        """
        if len(word) != len(guess):
            return False
            
        # Handle exact matches first (greens)
        remaining_word = list(word)
        remaining_guess = list(guess)
        
        # Remove green matches from consideration
        for i, (letter, clue) in enumerate(zip(guess, clues)):
            if clue == "Green" and word[i] != letter:
                return False
            if clue == "Green":
                remaining_word[i] = '_'
                remaining_guess[i] = '_'
                
        # Check yellows and grays
        for i, (letter, clue) in enumerate(zip(guess, clues)):
            if clue == "Green":
                continue
                
            if clue == "Yellow":
                # Letter must exist somewhere else
                if letter not in remaining_word:
                    return False
                # But not in this position
                if word[i] == letter:
                    return False
                # Remove the first occurrence for future yellow checks
                idx = remaining_word.index(letter)
                remaining_word[idx] = '_'
                
            if clue == "Gray":
                # Letter must not exist in remaining positions
                if letter in remaining_word:
                    return False
                    
        return True


# In[54]:


class WordleSolverLinus:
    def __init__(self, word_list: list[str]):
        """
        Initializes the solver with a list of possible words.
        """
        self.all_words = list(word_list)
        self.possible_words = list(word_list)
        self._letter_freq = self._calculate_letter_frequencies()

    def make_guess(self) -> str:
        """
        Returns the solver's next guess (a 5-letter string that must be a possible word in the list).
        Uses the internal state (e.g., possible_words) to make a guess.
        Returns None if no guess is possible.
        """
        if not self.possible_words:
            return None
            
        if len(self.possible_words) == 1:
            return self.possible_words[0]
            
        # Score each word based on letter frequencies and positions
        best_score = -1
        best_word = self.possible_words[0]
        
        for word in self.possible_words:
            score = 0
            seen = set()
            for i, letter in enumerate(word):
                if letter not in seen:
                    score += self._letter_freq[i].get(letter, 0)
                    seen.add(letter)
            if score > best_score:
                best_score = score
                best_word = word
                
        return best_word

    def submit_guess_and_get_clues(self, clues: list[str], guess: str):
        """
        Updates the solver's internal state based on the provided clues from a guess.
        """
        new_possible = []
        for word in self.possible_words:
            if self._matches_clues(word, guess, clues):
                new_possible.append(word)
        self.possible_words = new_possible
        self._letter_freq = self._calculate_letter_frequencies()

    def reset(self):
        """
        Resets the solver's internal state to start a new game.
        """
        self.possible_words = list(self.all_words)
        self._letter_freq = self._calculate_letter_frequencies()

    def _matches_clues(self, word: str, guess: str, clues: list[str]) -> bool:
        """
        Helper method to check if a word matches the clues from a guess.
        """
        # First check green positions
        for i, (clue, g_char) in enumerate(zip(clues, guess)):
            if clue == "Green" and word[i] != g_char:
                return False
                
        # Count remaining letters for yellow/gray checks
        remaining = {}
        for i, c in enumerate(word):
            if clues[i] != "Green" or guess[i] != c:
                remaining[c] = remaining.get(c, 0) + 1
                
        # Check yellows and grays
        for i, (clue, g_char) in enumerate(zip(clues, guess)):
            if clue == "Green":
                continue
            elif clue == "Yellow":
                if g_char not in remaining or remaining[g_char] == 0:
                    return False
                remaining[g_char] -= 1
            elif clue == "Gray" and g_char in remaining and remaining[g_char] > 0:
                return False
                
        return True
        
    def _calculate_letter_frequencies(self) -> dict:
        """
        Calculate letter frequencies for each position in remaining possible words.
        """
        freqs = [{} for _ in range(5)]
        if not self.possible_words:
            return freqs
            
        total = len(self.possible_words)
        for word in self.possible_words:
            for i, letter in enumerate(word):
                freqs[i][letter] = freqs[i].get(letter, 0) + 1
                
        # Convert to percentages
        for pos_freq in freqs:
            for letter in pos_freq:
                pos_freq[letter] = pos_freq[letter] / total
                
        return freqs


# In[41]:


class WordleSolverGuido:
    def __init__(self, word_list: list[str]):
        """
        Initializes the solver with a list of possible words.
        """
        self.original_words = word_list
        self.possible_words = word_list.copy()
        self.last_guess = None

    def make_guess(self) -> str:
        """
        Returns the solver's next guess (a 5-letter string that must be a possible word in the list).
        Uses the internal state (e.g., possible_words) to make a guess.
        Returns None if no guess is possible.
        """
        if not self.possible_words:
            return None
            
        if len(self.possible_words) == 1:
            return self.possible_words[0]
            
        # Score each word based on how well it would partition the remaining possibilities
        best_score = float('-inf')
        best_word = self.possible_words[0]
        
        # Use frequency analysis for early guesses when there are many possibilities
        if len(self.possible_words) > 100:
            letter_freq = {}
            for word in self.possible_words:
                for pos, letter in enumerate(word):
                    if (pos, letter) not in letter_freq:
                        letter_freq[(pos, letter)] = 0
                    letter_freq[(pos, letter)] += 1
                    
            word_scores = {}
            for word in self.possible_words:
                score = sum(letter_freq.get((pos, letter), 0) 
                          for pos, letter in enumerate(word))
                if score > best_score:
                    best_score = score
                    best_word = word
                    
        else:
            # For smaller sets, evaluate how well each guess splits remaining possibilities
            for guess in self.possible_words:
                score = self._evaluate_guess_score(guess)
                if score > best_score:
                    best_score = score
                    best_word = guess
                    
        self.last_guess = best_word
        return best_word

    def submit_guess_and_get_clues(self, clues: list[str], guess: str):
        """
        Updates the solver's internal state based on the provided clues from a guess.
        """
        if not clues or not guess:
            return
            
        new_possible = []
        for word in self.possible_words:
            if self._is_word_consistent(word, guess, clues):
                new_possible.append(word)
        self.possible_words = new_possible

    def reset(self):
        """
        Resets the solver's internal state to start a new game.
        """
        self.possible_words = self.original_words.copy()
        self.last_guess = None

    def _is_word_consistent(self, word: str, guess: str, clues: list[str]) -> bool:
        """
        Checks if a word is consistent with the clues from a guess.
        """
        if len(word) != len(guess):
            return False
            
        # Handle green clues first
        remaining_target_letters = list(word)
        remaining_guess_letters = list(guess)
        for i, (clue, guess_letter) in enumerate(zip(clues, guess)):
            if clue == "Green":
                if word[i] != guess_letter:
                    return False
                remaining_target_letters[i] = None
                remaining_guess_letters[i] = None
                
        # Handle yellow and gray clues
        for i, (clue, guess_letter) in enumerate(zip(clues, guess)):
            if remaining_guess_letters[i] is None:
                continue
                
            if clue == "Yellow":
                # Letter must exist somewhere else in remaining letters
                if guess_letter not in remaining_target_letters:
                    return False
                # Remove one instance of the letter
                idx = remaining_target_letters.index(guess_letter)
                remaining_target_letters[idx] = None
            elif clue == "Gray":
                # Letter should not exist in remaining letters
                if guess_letter in remaining_target_letters:
                    return False
                    
        return True

    def _evaluate_guess_score(self, guess: str) -> float:
        """
        Scores a potential guess based on how well it would partition remaining possibilities.
        """
        pattern_counts = {}
        total = len(self.possible_words)
        
        for word in self.possible_words:
            pattern = self._get_pattern(guess, word)
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
        # Calculate entropy score
        score = 0
        for count in pattern_counts.values():
            prob = count / total
            score -= prob * prob  # Higher score for more even splits
            
        return score

    def _get_pattern(self, guess: str, target: str) -> tuple:
        """
        Returns a tuple representing the pattern of greens/yellows/grays that would result
        from guessing 'guess' against 'target'.
        """
        result = []
        remaining_target = list(target)
        
        # First pass: mark greens
        for i, (g, t) in enumerate(zip(guess, target)):
            if g == t:
                result.append("Green")
                remaining_target[i] = None
            else:
                result.append(None)
                
        # Second pass: mark yellows and grays
        for i, (g, r) in enumerate(zip(guess, result)):
            if r is not None:
                continue
            if guess[i] in remaining_target:
                result[i] = "Yellow"
                idx = remaining_target.index(guess[i])
                remaining_target[idx] = None
            else:
                result[i] = "Gray"
                
        return tuple(result)


# In[47]:


class WordleSolverKatherine:
    def __init__(self, word_list: list[str]):
        """
        Initializes the solver with a list of possible words.
        """
        self.original_words = word_list.copy()
        self.possible_words = word_list.copy()
        self.letter_frequencies = self._calculate_letter_frequencies()
        
    def make_guess(self) -> str:
        """
        Returns the solver's next guess (a 5-letter string that must be a possible word in the list).
        Uses the internal state (e.g., possible_words) to make a guess.
        Returns None if no guess is possible.
        """
        if not self.possible_words:
            return None
            
        if len(self.possible_words) == 1:
            return self.possible_words[0]
            
        # Score each word based on letter frequency and position
        best_score = -1
        best_word = self.possible_words[0]
        
        for word in self.possible_words:
            score = 0
            used_letters = set()
            
            for i, letter in enumerate(word):
                if letter not in used_letters:
                    score += self.letter_frequencies[i].get(letter, 0)
                    used_letters.add(letter)
                    
            if score > best_score:
                best_score = score
                best_word = word
                
        return best_word

    def submit_guess_and_get_clues(self, clues: list[str], guess: str):
        """
        Updates the solver's internal state based on the provided clues from a guess.
        """
        # Create patterns based on clues
        must_contain = []
        must_not_contain = set()
        position_constraints = [set() for _ in range(5)]
        exact_positions = {}
        
        for i, (clue, letter) in enumerate(zip(clues, guess)):
            if clue == "Green":
                exact_positions[i] = letter
            elif clue == "Yellow":
                must_contain.append(letter)
                position_constraints[i].add(letter)
            else:  # Gray
                # Only add to must_not_contain if letter isn't in must_contain or exact_positions
                if (letter not in must_contain and 
                    letter not in exact_positions.values()):
                    must_not_contain.add(letter)
        
        # Filter possible words based on constraints
        new_possible_words = []
        
        for word in self.possible_words:
            valid = True
            
            # Check exact positions
            for pos, letter in exact_positions.items():
                if word[pos] != letter:
                    valid = False
                    break
                    
            if not valid:
                continue
                
            # Check position constraints
            for pos, letters in enumerate(position_constraints):
                if word[pos] in letters:
                    valid = False
                    break
                    
            if not valid:
                continue
                
            # Check must contain
            for letter in must_contain:
                if letter not in word:
                    valid = False
                    break
                    
            if not valid:
                continue
                
            # Check must not contain
            for letter in must_not_contain:
                if letter in word:
                    valid = False
                    break
                    
            if valid:
                new_possible_words.append(word)
                
        self.possible_words = new_possible_words
        self.letter_frequencies = self._calculate_letter_frequencies()

    def reset(self):
        """
        Resets the solver's internal state to start a new game.
        """
        self.possible_words = self.original_words.copy()
        self.letter_frequencies = self._calculate_letter_frequencies()
        
    def _calculate_letter_frequencies(self):
        """
        Calculate letter frequencies for each position in remaining possible words
        """
        frequencies = [{} for _ in range(5)]
        for word in self.possible_words:
            for i, letter in enumerate(word):
                frequencies[i][letter] = frequencies[i].get(letter, 0) + 1
        return frequencies


# In[60]:


class WordleSolverHybrid:
    def __init__(self, word_list: list[str]):
        """
        Initializes the solver with a list of possible words.
        """
        self.original_words = word_list
        self.possible_words = word_list.copy()
        self.letter_freqs = self._calculate_letter_frequencies()
        self.position_freqs = self._calculate_position_frequencies()
        
    def make_guess(self) -> str:
        """
        Returns the solver's next guess (a 5-letter string that must be a possible word in the list).
        Uses the internal state (e.g., possible_words) to make a guess.
        Returns None if no guess is possible.
        """
        if not self.possible_words:
            return None
            
        if len(self.possible_words) == 1:
            return self.possible_words[0]
            
        # Score each remaining word based on letter frequencies and positions
        best_score = float('-inf')
        best_word = self.possible_words[0]
        
        for word in self.possible_words:
            score = 0
            seen_letters = set()
            
            for i, letter in enumerate(word):
                if letter not in seen_letters:
                    score += self.letter_freqs.get(letter, 0)
                    score += self.position_freqs.get((letter, i), 0)
                    seen_letters.add(letter)
                else:
                    score *= 0.5  # Penalize repeated letters
                    
            if score > best_score:
                best_score = score
                best_word = word
                
        return best_word

    def submit_guess_and_get_clues(self, clues: list[str], guess: str):
        """
        Updates the solver's internal state based on the provided clues from a guess.
        """
        new_possible = []
        
        for word in self.possible_words:
            if self._is_word_consistent(word, guess, clues):
                new_possible.append(word)
                
        self.possible_words = new_possible
        self.letter_freqs = self._calculate_letter_frequencies()
        self.position_freqs = self._calculate_position_frequencies()

    def reset(self):
        """
        Resets the solver's internal state to start a new game.
        """
        self.possible_words = self.original_words.copy()
        self.letter_freqs = self._calculate_letter_frequencies()
        self.position_freqs = self._calculate_position_frequencies()

    def _calculate_letter_frequencies(self) -> dict:
        """Calculate frequency of each letter across all possible words"""
        freqs = {}
        for word in self.possible_words:
            for letter in set(word):  # Count each letter once per word
                freqs[letter] = freqs.get(letter, 0) + 1
        return freqs
        
    def _calculate_position_frequencies(self) -> dict:
        """Calculate frequency of each letter in each position"""
        freqs = {}
        for word in self.possible_words:
            for i, letter in enumerate(word):
                freqs[(letter, i)] = freqs.get((letter, i), 0) + 1
        return freqs

    def _is_word_consistent(self, word: str, guess: str, clues: list[str]) -> bool:
        """Check if a word is consistent with the clues from a guess"""
        word_letters = list(word)
        guess_letters = list(guess)
        
        # First check green clues (exact matches)
        for i, (clue, guess_letter) in enumerate(zip(clues, guess_letters)):
            if clue == "Green" and word_letters[i] != guess_letter:
                return False
            if clue == "Green":
                word_letters[i] = None  # Mark as used
                guess_letters[i] = None  # Mark as matched
                
        # Then check yellow clues (letter in wrong position)
        for i, (clue, guess_letter) in enumerate(zip(clues, guess_letters)):
            if clue == "Yellow":
                if guess_letter not in word_letters:  # Letter must be in remaining letters
                    return False
                if word_letters[i] == guess_letter:  # Letter can't be in this position
                    return False
                word_letters[word_letters.index(guess_letter)] = None  # Mark as used
                
        # Finally check gray clues
        for i, (clue, guess_letter) in enumerate(zip(clues, guess_letters)):
            if clue == "Gray" and guess_letter in word_letters:
                return False
                
        return True


# In[64]:


# Example solvers to evaluate
solvers = [
    ("No Person 1", WordleSolverNoPers1),
    ("No Person 2", WordleSolverNoPers2),
    ("No Person 3", WordleSolverNoPers3),
    ("No Person 4", WordleSolverNoPers4),    
    ("Carmack", WordleSolverCarmack),
    ("Knuth", WordleSolverKnuth),
    ("Katherine", WordleSolverKatherine),
    ("Fowler", WordleSolverFowler),
    ("Linus", WordleSolverLinus),
    ("Torvalds", WordleSolverTorvalds),
    ("Guido", WordleSolverGuido),
    ("Hybrid", WordleSolverHybrid),
]

# Initialize the performance records list
performance_records = []

for personality_name, solver_class in solvers:
    win_rate, avg_attempts, avg_guess_time = WordleSimulator.simulate_games(
        solver_class,
        word_list,
        num_games=NUM_TEST_GAMES,
        max_attempts=MAX_ATTEMPTS,
        secret_words=fixed_secret_words
    )
    # Create a performance record for the current solver
    performance = SolverPerformance(
        personality_name=personality_name,
        win_rate=win_rate,
        avg_attempts=avg_attempts,
        avg_guess_time=avg_guess_time
    )
    # Add the record to the list
    performance_records.append(performance)

# Display the collected performance data
for record in performance_records:
    print(f"{record.personality_name} Results:")
    print(f"  Win Rate: {record.win_rate * 100:.1f}%")
    print(f"  Avg Attempts (Wins Only): {record.avg_attempts:.2f}")
    print(f"  Avg Guess Time: {record.avg_guess_time * 1000:.3f} ms")
    print("")


# In[ ]:




