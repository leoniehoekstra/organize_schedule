#!/usr/bin/env python3
"""Analyze workshop oversubscription and random assignments.

This script reads the workshop schedule, student preferences and the final
assignments. For each workshop it reports capacity, how many students
ranked it first or second, and how many were actually assigned by rank
category. The output highlights workshops where demand exceeded capacity
and where random assignments occurred.
"""
import pandas as pd
from collections import Counter, defaultdict

PREFS = 'student_preferences_long_v5.csv'
FINAL = 'final_workshop_fixed.csv'
SCHEDULE = 'workshop_schedule.csv'


def load_data():
    sched = pd.read_csv(SCHEDULE)
    prefs = pd.read_csv(PREFS)
    final = pd.read_csv(FINAL)
    return sched, prefs, final


def summarize():
    sched, prefs, final = load_data()

    # capacity per workshop title (sum over sessions)
    capacity = sched.groupby('Workshop Title')['Capacity'].sum()

    # count of preferences at rank 1 and 2
    rank_counts = prefs[prefs['Rank'].isin([1, 2])]
    pref_counts = rank_counts.groupby(['Workshop', 'Rank']).size().unstack(fill_value=0)

    # lookup for student->rank for each workshop
    rank_lookup = {
        (row.Student, row.Workshop): row.Rank for row in prefs.itertuples()
    }

    assigned_counts = defaultdict(Counter)
    for row in final.itertuples():
        wk = row._5  # Workshop Title column
        st = row.Student
        r = rank_lookup.get((st, wk))
        if r == 1:
            cat = 'first'
        elif r == 2:
            cat = 'second'
        else:
            cat = 'random'
        assigned_counts[wk][cat] += 1

    summary_rows = []
    workshops = sorted(capacity.index)
    for w in workshops:
        cap = int(capacity.get(w, 0))
        first_pref = int(pref_counts.loc[w][1]) if w in pref_counts.index and 1 in pref_counts.loc[w] else 0
        second_pref = int(pref_counts.loc[w][2]) if w in pref_counts.index and 2 in pref_counts.loc[w] else 0
        demand = first_pref + second_pref
        first_ass = assigned_counts[w]['first']
        second_ass = assigned_counts[w]['second']
        random_ass = assigned_counts[w]['random']
        summary_rows.append({
            'Workshop': w,
            'Capacity': cap,
            'PrefFirst': first_pref,
            'PrefSecond': second_pref,
            'Demand': demand,
            'AssignedFirst': first_ass,
            'AssignedSecond': second_ass,
            'AssignedRandom': random_ass,
        })

    df = pd.DataFrame(summary_rows)
    df['OverSubscribed'] = df['Demand'] - df['Capacity']
    df.sort_values(['OverSubscribed', 'AssignedRandom'], ascending=False, inplace=True)

    pd.set_option('display.max_rows', None)
    print(df.to_string(index=False))


if __name__ == '__main__':
    summarize()
