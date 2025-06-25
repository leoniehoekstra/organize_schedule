#!/usr/bin/env python3
"""Classify final workshop assignments by preference rank.

This script reads `student_preferences_long_v5.csv` and
`final_workshop_fixed.csv` from the current directory.
It counts how many assignments match first choices, second choices or
were not ranked ("random").  It also groups students by whether all of
their assignments were first choices, if they had any second choices
but no random assignments, or if they received at least one random
assignment.
"""
import csv
from collections import Counter, defaultdict

PREFS_FILE = 'student_preferences_long_v5.csv'
FINAL_FILE = 'final_workshop_fixed.csv'


def load_preferences(filename=PREFS_FILE):
    """Return mapping (student, zone, workshop) -> lowest rank."""
    mapping = {}
    with open(filename, encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            student = row['Student'].strip().lower()
            zone = row['Zone'].strip()
            workshop = row['Workshop'].strip()
            try:
                rank = int(row['Rank'])
            except ValueError:
                continue
            key = (student, zone, workshop)
            if key not in mapping or rank < mapping[key]:
                mapping[key] = rank
    return mapping


def classify_assignments(prefs, filename=FINAL_FILE):
    """Classify assignments and group students by category."""
    counts = Counter()
    per_student = defaultdict(list)

    with open(filename, encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            student = row['Student'].strip().lower()
            zone = row['Zone'].strip()
            workshop = row['Workshop Title'].strip()
            rank = prefs.get((student, zone, workshop))
            if rank == 1:
                cat = 'first'
            elif rank == 2:
                cat = 'second'
            else:
                cat = 'random'
            counts[cat] += 1
            per_student[student].append(cat)

    categories = {'first': [], 'second': [], 'random': []}
    for student, cats in per_student.items():
        if all(c == 'first' for c in cats):
            categories['first'].append(student)
        elif 'random' in cats:
            categories['random'].append(student)
        else:
            categories['second'].append(student)

    for lst in categories.values():
        lst.sort()

    return counts, categories


def main():
    prefs = load_preferences()
    counts, cats = classify_assignments(prefs)

    print("Assignments per category:")
    for cat in ['first', 'second', 'random']:
        print(f"  {cat:6}: {counts.get(cat, 0)}")

    print("\nStudents per category:")
    for cat in ['first', 'second', 'random']:
        names = ', '.join(cats[cat])
        print(f"  {cat:6} ({len(cats[cat])}): {names}")


if __name__ == '__main__':
    main()
