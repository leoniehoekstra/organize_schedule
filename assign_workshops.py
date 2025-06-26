#!/usr/bin/env python3
"""
assign_workshops.py

Assign students to workshops by minimizing sum of preference ranks,
respecting capacities and full-day exceptions, via PuLP.
"""

import pandas as pd
import pulp

def load_data():
    sched = pd.read_csv(
        'workshop_schedule.csv',
        dtype={'Session': int, 'Full_Day_Session': int, 'Capacity': int}
    )
    prefs = pd.read_csv(
        'student_preferences_long_v5.csv',
        dtype={'Rank': int}
    )
    return sched, prefs

def build_zone_map(prefs):
    return prefs.groupby('Workshop')['Zone'].first().to_dict()

def build_costs(prefs):
    cost = {}
    for r in prefs.itertuples(index=False):
        key = (r.Student, r.Workshop)
        cost[key] = min(r.Rank, cost.get(key, r.Rank))
    return cost

def main():
    # 1) Load everything
    sched, prefs = load_data()
    zone_map     = build_zone_map(prefs)
    cost         = build_costs(prefs)

    # 2) Define exactly‐preassigned slots for Jesse & Niels
    pre_assign = {
        'jesse wolters': [
            ('Vissen', 'Tuesday',       1),
            ('Papier leer', 'Tuesday',  2),
            ('Hond in de hoofdrol','Wednesday',1),
            ('Vuvuzela maken', 'Wednesday',2),
            ('Naar de kaasboerderij', 'Thursday',0),
        ],
        'niels hielkema': [
            ('Vissen', 'Tuesday',       1),
            ('Papier leer', 'Tuesday',  2),
            ('Hond in de hoofdrol','Wednesday',1),
            ('Vuvuzela maken', 'Wednesday',2),
            ('Naar de kaasboerderij', 'Thursday',0),
        ],
    }
    forced_students = set(pre_assign.keys())

    # 3) Build ws_specs and immediately subtract preassign seats
    grp = (
        sched
        .groupby(['Day','Workshop Title','Session','Full_Day_Session'])['Capacity']
        .sum()
        .reset_index()
    )
    days = sorted(grp['Day'].unique())

    ws_specs = []
    for _, row in grp.iterrows():
        w = row['Workshop Title']
        t = int(row['Session'])
        d = row['Day']
        full = bool(int(row['Full_Day_Session']))
        cap = int(row['Capacity'])
        # subtract one seat for each forced student sitting here
        for student, slots in pre_assign.items():
            if (w, d, t) in slots:
                cap -= 1
        ws_specs.append((w, t, d, cap, full))
    #print
    workshops = sorted({ w for (w, t, d, cap, full) in ws_specs })

    # dump it to the console, showing repr() so you can spot stray spaces:
    print("=== Unique workshop titles ===")
    for w in workshops:
        print(f"– {w!r}")
    print("=============================")

    # 4) Our solver will only see the *other* students
    students = [
        s for s in sorted(prefs['Student'].unique())
        if s not in forced_students
    ]

    # 5) Build the LP
    prob = pulp.LpProblem('Workshop_Assignment', pulp.LpMinimize)
    x = {}
    for s in students:
        for (w, t, d, cap, full) in ws_specs:
            name = f"x_{s}_{w.replace(' ','_')}_{d}_T{t}"
            x[(s, w, d, t)] = pulp.LpVariable(name, cat='Binary')

    # 6) Two‐per‐zone (full‐day counts as 2)
    zones = sorted(prefs['Zone'].unique())
    # 6) Two‐per‐zone with explicit first/second logic
    for s in students:
        for z in zones:
            half_pairs = [
                (w, x[(s,w,d,t)])
                for (w,t,d,cap,full) in ws_specs
                if not full and zone_map[w] == z
            ]
            full_pairs = [
                (w, x[(s,w,d,0)])
                for (w,t,d,cap,full) in ws_specs
                if full     and zone_map[w] == z
            ]

            first_half  = [var for (w,var) in half_pairs  if cost.get((s,w), 999) == 1]
            first_full  = [var for (w,var) in full_pairs  if cost.get((s,w), 999) == 1]
            second_half = [var for (w,var) in half_pairs  if cost.get((s,w), 999) == 2]
            second_full = [var for (w,var) in full_pairs  if cost.get((s,w), 999) == 2]
            other_half  = [var for (w,var) in half_pairs  if cost.get((s,w), 999) not in (1,2)]
            other_full  = [var for (w,var) in full_pairs  if cost.get((s,w), 999) not in (1,2)]

            fsz = pulp.lpSum(first_half ) + 2 * pulp.lpSum(first_full)
            ssz = pulp.lpSum(second_half) + 2 * pulp.lpSum(second_full)
            osz = pulp.lpSum(other_half) + 2 * pulp.lpSum(other_full)

            # determine how many random workshops may be used
            pref_set = {
                w
                for (w, _) in half_pairs + full_pairs
                if cost.get((s, w), 999) in (1, 2)
            }
            allow_random = max(0, 2 - len(pref_set))

            prob += (fsz + ssz + osz == 2,     f"TwoPerZone_{s}_{z}")
            prob += (ssz >= 2 - fsz,           f"UseSeconds_{s}_{z}")
            prob += (osz <= allow_random,      f"RandLimit_{s}_{z}")

    # 7) Capacity
    for (w,t,d,cap,full) in ws_specs:
        prob += (
            pulp.lpSum(x[(s,w,d,t)] for s in students) <= cap,
            f"Cap_{w.replace(' ','_')}_{d}_T{t}"
        )

    # 8) No repeats – only once per (student, workshop)
    workshops = sorted({w for (w,_,_,_,_) in ws_specs})
    for s in students:
        for w0 in workshops:
            prob += (
                pulp.lpSum(
                    x[(s,w0,d2,t2)]
                    for (w1,t2,d2,cap2,full2) in ws_specs
                    if w1 == w0
                ) <= 1,
                f"NoRepeat_{s}_{w0.replace(' ','_')}"
            )



    # 9) One slot per student per day/session
    for s in students:
        for d in days:
            # session 1
            prob += (
                pulp.lpSum(
                    x[(s,w,d,1)]
                    for (w,t,d2,cap,full) in ws_specs
                    if d2==d and t==1
                )
                + pulp.lpSum(
                    x[(s,w,d,0)]
                    for (w,t,d2,cap,full) in ws_specs
                    if d2==d and t==0 and full
                ) == 1,
                f"OnePerSlot_{s}_{d}_Sess1"
            )
            # session 2
            prob += (
                pulp.lpSum(
                    x[(s,w,d,2)]
                    for (w,t,d2,cap,full) in ws_specs
                    if d2==d and t==2
                )
                + pulp.lpSum(
                    x[(s,w,d,0)]
                    for (w,t,d2,cap,full) in ws_specs
                    if d2==d and t==0 and full
                ) == 1,
                f"OnePerSlot_{s}_{d}_Sess2"
            )

    # 10) Objective
    prob += pulp.lpSum(
        cost.get((s,w), max(prefs['Rank'])+1) * var
        for ((s,w,d,t),var) in x.items()
    )

    # 11) Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=True, timeLimit=60))

    # 12) Extract forced + optimized assignments
    rows = []
    # first the two manual ones:
    for student, slots in pre_assign.items():
        for (w,d,t) in slots:
            rows.append({
                'Student': student,
                'Zone':    zone_map[w],
                'Day':     d,
                'Session': t,
                'Workshop Title': w
            })
    # then the solver output:
    for (s,w,d,t),var in x.items():
        if var.value()==1:
            rows.append({
                'Student': s,
                'Zone':    zone_map[w],
                'Day':     d,
                'Session': t,
                'Workshop Title': w
            })

    out = pd.DataFrame(rows)
    out = out[['Student','Zone','Day','Session','Workshop Title']]

    # right before out.to_csv(...)
    df = pd.DataFrame(rows)
    dups = (
        df
        .groupby(['Student','Workshop Title'])
        .size()
        .reset_index(name='count')
        .query('count > 1')
    )
    if not dups.empty:
        print("Found duplicate assignments!")
        print(dups)


    out.to_csv('final_workshop_fixed.csv', index=False)

    print("Solved: status =", pulp.LpStatus[prob.status])
    print("Assignments saved to final_workshop_fixed.csv")

if __name__ == '__main__':
    main()
