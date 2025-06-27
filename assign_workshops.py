"""
assign_workshops.py

Assign students to workshops by minimizing sum of preference ranks,
respecting capacities and full-day exceptions, via PuLP.
"""

import pandas as pd
import pulp
from datetime import datetime

def load_data():
    sched = pd.read_csv(
        'workshop_schedule.csv',
        dtype={'Session': int, 'Full_Day_Session': int, 'Capacity': int}
    )
    prefs = pd.read_csv(
        'student_preferences_long_v8.csv',
        dtype={'Rank': int},
    )
    # parse submission dates so we can order students chronologically
    prefs['Parsed_Date'] = pd.to_datetime(
        prefs['Date'], format='%d-%m-%Y %H:%M:%S'
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

def build_student_dates(prefs):
    """Return a mapping from student name to submission datetime."""
    return prefs.groupby('Student')['Parsed_Date'].first().to_dict()


def solve_group(students, zone_map, cost, cap_map, full_map, days, *, late: bool = False):
    """Solve the optimization for a subset of students.

    Parameters
    ----------
    students : list[str]
        Students to schedule in this run.
    zone_map : dict[str, str]
        Mapping from workshop to zone.
    cost : dict[tuple[str, str], int]
        Mapping (student, workshop) -> rank cost.
    cap_map : dict[tuple[str, str, int], int]
        Remaining capacity for each (workshop, day, session).
    full_map : dict[tuple[str, str, int], bool]
        Whether the slot is a full day session.
    days : list[str]
        Ordered list of days in the schedule.

    Returns
    -------
    list[dict]
        Rows describing the assignments for the provided students.  The
        ``cap_map`` will be updated in place.
    """

    if not students:
        return []

    prob = pulp.LpProblem('Workshop_Assignment', pulp.LpMinimize)
    x = {
        (s, w, d, t): pulp.LpVariable(
            f"x_{s}_{w.replace(' ','_')}_{d}_T{t}", cat='Binary'
        )
        for s in students
        for (w, d, t) in cap_map
    }

    zones = sorted(set(zone_map.values()))

    # ------------------------------------------------------------------
    # Two per zone with first→second→random fallback
    # ------------------------------------------------------------------
    for s in students:
        for z in zones:
            half_pairs = [
                (w, x[(s,w,d,t)])
                for (w,d,t) in cap_map
                if not full_map[(w,d,t)] and zone_map[w] == z
            ]
            full_pairs = [
                (w, x[(s,w,d,0)])
                for (w,d,t) in cap_map
                if     full_map[(w,d,t)] and zone_map[w] == z
            ]

            # collect distinct first‑ and second‑choice **workshop titles**
            rank1_set = { w for (w,_) in half_pairs+full_pairs
                        if cost.get((s,w),999)==1 }
            rank2_set = { w for (w,_) in half_pairs+full_pairs
                        if cost.get((s,w),999)==2 and w not in rank1_set }

            # bucket the vars by preference tier
            first_half  = [v for (w,v) in half_pairs if w in rank1_set]
            first_full  = [v for (w,v) in full_pairs if w in rank1_set]
            second_half = [v for (w,v) in half_pairs if w in rank2_set]
            second_full = [v for (w,v) in full_pairs if w in rank2_set]
            other_half  = [v for (w,v) in half_pairs
                            if w not in rank1_set and w not in rank2_set]
            other_full  = [v for (w,v) in full_pairs
                            if w not in rank1_set and w not in rank2_set]

            fsz = pulp.lpSum(first_half)  + 2*pulp.lpSum(first_full)
            ssz = pulp.lpSum(second_half) + 2*pulp.lpSum(second_full)
            osz = pulp.lpSum(other_half)  + 2*pulp.lpSum(other_full)

            # how many random slots are allowed?
            # only if no distinct second choices were provided
            allow_random = 1 if len(rank2_set)==0 else 0

            # 1) exactly two slots in this zone
            prob += (fsz + ssz + osz == 2,
                    f"TwoPerZone_{s}_{z}")

            # 2) fill any missing #1 slots with #2s
            prob += (ssz >= 2 - fsz,
                    f"UseSeconds_{s}_{z}")

            # 3) if they supplied *no* distinct 2nd choices, allow up to one wild‑card
            prob += (osz <= allow_random,
                    f"RandLimit_{s}_{z}")
    # ------------------------------------------------------------------

    # Capacity constraints
    for (w, d, t), cap in cap_map.items():
        prob += (
            pulp.lpSum(x[(s, w, d, t)] for s in students) <= cap,
            f"Cap_{w.replace(' ','_')}_{d}_T{t}"
        )

    # # ── No repeats: each student may take a given workshop at most once ──
    # workshops = sorted({ w for (w, d, t) in cap_map.keys() })
    # for s in students:
    #     for w0 in workshops:
    #         # collect all decision vars for this student & workshop
    #         terms = []
    #         for (s2, w2, d2, t2), var in x.items():
    #             if s2 == s and w2 == w0:
    #                 terms.append(var)
    #         # enforce at most one assignment to workshop w0 for student s
    #         prob += (
    #             pulp.lpSum(terms) <= 1,
    #             f"NoRepeat_{s}_{w0.replace(' ','_')}"
    #         )
    #     # ── No repeats: each student may take a given workshop at most once ──
        # ── No repeats: each student may take a given workshop at most once ──
    workshops = sorted({ w for (w, d, t) in cap_map.keys() })
    for s in students:
        for w in workshops:
            prob += (
                pulp.lpSum(
                    x[(s, w, d, t)]
                    for (w2, d, t) in cap_map
                    if w2 == w
                ) <= 1,
                f"NoRepeat_{s}_{w.replace(' ','_')}"
            )




    # One slot per day/session
    for s in students:
        for d in days:
            prob += (
                pulp.lpSum(
                    x[(s, w, d, 1)]
                    for (w, d2, t) in cap_map
                    if d2 == d and t == 1
                )
                + pulp.lpSum(
                    x[(s, w, d, 0)]
                    for (w, d2, t) in cap_map
                    if d2 == d and t == 0 and full_map[(w, d2, t)]
                )
                == 1,
                f"OnePerSlot_{s}_{d}_Sess1",
            )
            prob += (
                pulp.lpSum(
                    x[(s, w, d, 2)]
                    for (w, d2, t) in cap_map
                    if d2 == d and t == 2
                )
                + pulp.lpSum(
                    x[(s, w, d, 0)]
                    for (w, d2, t) in cap_map
                    if d2 == d and t == 0 and full_map[(w, d2, t)]
                )
                == 1,
                f"OnePerSlot_{s}_{d}_Sess2",
            )

    # Objective
    prob += pulp.lpSum(
        cost.get((s, w), 99) * var
        for ((s, w, d, t), var) in x.items()
    )

    prob.solve(pulp.PULP_CBC_CMD(msg=True, timeLimit=60))

    rows = []
    for (s, w, d, t), var in x.items():
        if var.value() == 1:
            rows.append({
                'Student': s,
                'Zone': zone_map[w],
                'Day': d,
                'Session': t,
                'Workshop Title': w,
            })
            cap_map[(w, d, t)] -= 1

    return rows

def main():
    # 1) Load everything
    sched, prefs = load_data()
    zone_map = build_zone_map(prefs)
    cost = build_costs(prefs)
    dates = build_student_dates(prefs)

    # 2) Define exactly-preassigned slots for Jesse & Niels
    pre_assign = {
        'jesse wolters': [
            ('Vissen', 'Tuesday', 1),
            ('Papier leer', 'Tuesday', 2),
            ('Hond in de hoofdrol', 'Wednesday', 1),
            ('Vuvuzela maken', 'Wednesday', 2),
            ('Naar de kaasboerderij', 'Thursday', 0),
        ],
        'niels hielkema': [
            ('Vissen', 'Tuesday', 1),
            ('Papier leer', 'Tuesday', 2),
            ('Hond in de hoofdrol', 'Wednesday', 1),
            ('Vuvuzela maken', 'Wednesday', 2),
            ('Naar de kaasboerderij', 'Thursday', 0),
        ],
    }

    forced_students = set(pre_assign.keys())

    # 3) Build capacity maps and subtract preassigned seats
    grp = (
        sched
        .groupby(['Day', 'Workshop Title', 'Session', 'Full_Day_Session'])['Capacity']
        .sum()
        .reset_index()
    )

    days = sorted(grp['Day'].unique())
    cap_map = {}
    full_map = {}
    for _, row in grp.iterrows():
        w = row['Workshop Title']
        t = int(row['Session'])
        d = row['Day']
        full = bool(int(row['Full_Day_Session']))
        cap = int(row['Capacity'])
        for student, slots in pre_assign.items():
            if (w, d, t) in slots:
                cap -= 1
        cap_map[(w, d, t)] = cap
        full_map[(w, d, t)] = full

    # 4) Determine student groups based on submission date
    early_cut = datetime(2025, 6, 23)
    mid_cut = datetime(2025, 6, 24)

    all_students = [s for s in sorted(prefs['Student'].unique()) if s not in forced_students]
    students_early = [s for s in all_students if dates[s] < early_cut]
    students_mid = [s for s in all_students if early_cut <= dates[s] < mid_cut]
    students_late = [s for s in all_students if dates[s] >= mid_cut]

    rows = []
    for student, slots in pre_assign.items():
        for (w, d, t) in slots:
            rows.append({
                'Student': student,
                'Zone': zone_map[w],
                'Day': d,
                'Session': t,
                'Workshop Title': w,
            })

    # 5) Solve sequentially for each group
    # rows += solve_group(students_early, zone_map, cost, cap_map, full_map, days)
    # rows += solve_group(students_mid, zone_map, cost, cap_map, full_map, days)
    # rows += solve_group(students_late, zone_map, cost, cap_map, full_map, days)
    print("\n🩺  Feasibility check ‑ early cohort one‑by‑one")
    for stu in students_early:
        # make an isolated copy of the capacity map for this test
        cap_tmp = cap_map.copy()

        # build a *tiny* problem for a single student
        try:
            rows_test = solve_group(
                [stu],          # only this student
                zone_map,
                cost,
                cap_tmp,
                full_map,
                days,
            )
            if not rows_test:          # no rows returned  -> infeasible
                print(f"   ✘ fails for {stu!r}")
            else:                      # at least one assignment -> feasible
                print(f"   ✓ ok for   {stu!r}")
        except pulp.PulpSolverError:
            # CBC raised an error instead of returning normally
            print(f"   ✘ fails for {stu!r} (solver error)")
    print("🩺  Feasibility probe finished\n")

    # right after building cap_map:
    print("\n--- Thursday full‑day sessions ---")
    print(pd.DataFrame([
        {'Workshop': w, 'Day': d, 'Session': t, 'Cap': cap}
        for (w, d, t), cap in cap_map.items()
        if d == 'Thursday'
    ]))
    print("\n--- Wednesday half‑day for Zwemmen ---")
    print(cap_map.get(('Zwemmen','Wednesday',1), '<missing>'),
        cap_map.get(('Zwemmen','Wednesday',2), '<missing>'))
    print("\n--- Wednesday half‑day for Vissen ---")
    print(cap_map.get(('Vissen','Wednesday',1), '<missing>'),
        cap_map.get(('Vissen','Wednesday',2), '<missing>'))


    print("\n🩺  Incremental build‑up test (early cohort)")
    cap_tmp = cap_map.copy()
    chosen  = []

    for stu in students_early:
        chosen.append(stu)
        try:
            # Solve for everyone chosen so far, but on a *copy* of cap_tmp
            rows_tmp = solve_group(chosen,
                                zone_map,
                                cost,
                                cap_tmp.copy(),
                                full_map,
                                days,
                                late=False)
            if rows_tmp:
                print(f"   ✓ cumulative ok with {len(chosen):3} students (last added: {stu})")
                # Only remove seats for *this* new student
                new_assigns = [r for r in rows_tmp if r['Student'] == stu]
                for r in new_assigns:
                    key = (r['Workshop Title'], r['Day'], r['Session'])
                    cap_tmp[key] -= 1
            else:
                print(f"   ✘ infeasible when adding {stu!r} (student #{len(chosen)})")
                break

        except pulp.PulpSolverError:
            print(f"   ✘ solver error when adding {stu!r} (student #{len(chosen)})")
            break

    print("🩺  Incremental test finished\n")



    rows += solve_group(students_early, zone_map, cost, cap_map, full_map, days, late=False)

    rows += solve_group(students_mid, zone_map, cost, cap_map, full_map, days, late=False)

    rows += solve_group(students_late,  zone_map, cost, cap_map, full_map, days, late=True)


    out = pd.DataFrame(rows)
    out = out[['Student', 'Zone', 'Day', 'Session', 'Workshop Title']]

    df = pd.DataFrame(rows)
    dups = (
        df
        .groupby(['Student', 'Workshop Title'])
        .size()
        .reset_index(name='count')
        .query('count > 1')
    )
    if not dups.empty:
        print('Found duplicate assignments!')
        print(dups)

    out.to_csv('FINAL_workshop_schedule_v1.csv', index=False)

    print('Solved sequentially. Assignments saved to FINAL_workshop_schedule_v1.csv')


if __name__ == '__main__':
    main()
