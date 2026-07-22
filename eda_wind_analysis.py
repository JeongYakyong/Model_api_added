"""
Wind Farm EDA — East/West zone operational analysis
Usage: python eda_wind_analysis.py
"""
import pandas as pd
import numpy as np

# ── Load ─────────────────────────────────────────────────────────────────────
df = pd.read_csv('eda_wind.csv', encoding='cp949')
df.columns = ['idx', 'timestamp', 'wind_west', 'wind_east',
              'wdir_west', 'wdir_east', 'real_wind_gen',
              'Wind_Utilization', 'Wind_Capacity_Est']
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp').sort_index()
df = df.dropna(subset=['wind_west', 'wind_east', 'Wind_Capacity_Est'])

# ── Capacity-based time-varying east/west weights ─────────────────────────────
# Post-expansion (2025-02+) capacity is stable: 363.9 MW (rolling max)
# East installed = 201.1 MW (unchanged throughout)
# Capture rate = 363.9 / 418.9 = 86.9%  →  East rolling max ≈ 174.8 MW (fixed)
EAST_ROLLMAX   = 174.8          # MW — east contribution to rolling max (constant)
df['west_cap'] = (df['Wind_Capacity_Est'] - EAST_ROLLMAX).clip(lower=1.0)
df['W_east']   = EAST_ROLLMAX / df['Wind_Capacity_Est']
df['W_west']   = df['west_cap'] / df['Wind_Capacity_Est']

# ── Capacity timeline ─────────────────────────────────────────────────────────
print("=" * 60)
print("1. Wind Capacity (rolling max) — monthly")
print("=" * 60)
monthly = df['Wind_Capacity_Est'].resample('ME').last()
for ts, v in monthly.items():
    w_w = (v - EAST_ROLLMAX) / v
    w_e = EAST_ROLLMAX / v
    print(f"  {ts.strftime('%Y-%m')}:  {v:6.1f} MW   W_west={w_w:.3f}  W_east={w_e:.3f}")

# ── Define periods ────────────────────────────────────────────────────────────
# Pre-expansion  : Jan–Sep 2024  (capacity ~258 MW, west≈32%, east≈68%)
# Transition     : Oct 2024–Jan 2025 (new turbines spinning up — exclude)
# Post-expansion : Feb 2025+    (capacity stable ~364 MW, west≈52%, east≈48%)
pre  = df[df.index <  '2024-10-01'].copy()
post = df[df.index >= '2025-02-01'].copy()

CUT_IN = 3.0

def add_state(d):
    d = d.copy()
    d['op_west'] = (d['wind_west'] >= CUT_IN).astype(float)
    d['op_east'] = (d['wind_east'] >= CUT_IN).astype(float)
    d['eff']     = d['W_west'] * d['op_west'] + d['W_east'] * d['op_east']
    d['blend']   = d['W_west'] * d['wind_west'] + d['W_east'] * d['wind_east']
    d['blend_op']= (d['blend'] >= CUT_IN).astype(float)
    d['state']   = d.apply(
        lambda r: 'both_on'   if r['op_west']==1 and r['op_east']==1 else
                  'west_only' if r['op_west']==1 and r['op_east']==0 else
                  'east_only' if r['op_west']==0 and r['op_east']==1 else
                  'both_off', axis=1)
    # Distortion: blend says generation happening but one zone is offline
    d['distorted'] = (d['blend_op'] == 1) & (d['eff'] < 0.99)
    return d

pre  = add_state(pre)
post = add_state(post)

# ── State distribution ────────────────────────────────────────────────────────
print()
print("=" * 60)
print("2. Operational state distribution")
print("=" * 60)
for label, d in [('Pre-expansion (Jan–Sep 2024)', pre),
                 ('Post-expansion (Feb 2025+)',   post)]:
    print(f"\n  [{label}]")
    print(f"  W_west avg={d['W_west'].mean():.3f}  W_east avg={d['W_east'].mean():.3f}")
    n = len(d)
    vc = d['state'].value_counts()
    for s in ['both_on', 'west_only', 'east_only', 'both_off']:
        c = vc.get(s, 0)
        print(f"    {s:12s}: {c:5d} hrs ({c/n*100:5.1f}%)")
    dist = d['distorted'].sum()
    print(f"    Distorted    : {dist:5d} hrs ({dist/n*100:5.1f}%)")

# ── Mean utilization by state ─────────────────────────────────────────────────
print()
print("=" * 60)
print("3. Mean Wind_Utilization by state")
print("=" * 60)
for label, d in [('Pre', pre), ('Post', post)]:
    print(f"\n  [{label}]")
    g = d.groupby('state')['Wind_Utilization'].agg(['mean', 'median', 'count'])
    print(g.round(3).to_string())

# ── Distortion analysis ───────────────────────────────────────────────────────
print()
print("=" * 60)
print("4. Distortion: how much does one offline zone inflate blend?")
print("=" * 60)
for label, d in [('Pre', pre), ('Post', post)]:
    dist = d[d['distorted']].copy()
    if dist.empty:
        continue
    print(f"\n  [{label}] — {len(dist)} distorted hours")
    print(f"    Mean effective fraction : {dist['eff'].mean():.3f}")
    print(f"    Mean blend wind speed   : {dist['blend'].mean():.2f} m/s")
    # What correction factor would apply
    print(f"    Correction factor range : {dist['eff'].min():.3f} – {dist['eff'].max():.3f}")
    by_state = dist.groupby('state').agg(
        hours=('eff','count'),
        mean_eff=('eff','mean'),
        mean_util=('Wind_Utilization','mean'),
        mean_west=('wind_west','mean'),
        mean_east=('wind_east','mean'),
    )
    print(by_state.round(3).to_string())

# ── Soft ramp calibration ─────────────────────────────────────────────────────
print()
print("=" * 60)
print("5. Actual generation near cut-in (west offline, east operational)")
print("   → calibrate soft ramp width")
print("=" * 60)
for label, d in [('Pre', pre), ('Post', post)]:
    print(f"\n  [{label}]  (east >= 3m/s, focus on west ramp zone)")
    sub = d[d['op_east'] == 1].copy()
    for lo, hi in [(0,1),(1,2),(2,2.5),(2.5,3),(3,3.5),(3.5,4),(4,5),(5,7),(7,15)]:
        s = sub[(sub['wind_west'] >= lo) & (sub['wind_west'] < hi)]
        if len(s) < 10:
            continue
        print(f"    west {lo:4.1f}–{hi:4.1f} m/s: "
              f"n={len(s):4d}  util={s['Wind_Utilization'].mean():.3f}  "
              f"as_frac_of_max={s['Wind_Utilization'].mean()/s['W_west'].mean():.3f}")

print()
print("  (west >= 3m/s, focus on east ramp zone)")
for label, d in [('Pre', pre), ('Post', post)]:
    print(f"\n  [{label}]")
    sub = d[d['op_west'] == 1].copy()
    for lo, hi in [(0,1),(1,2),(2,2.5),(2.5,3),(3,3.5),(3.5,4),(4,5),(5,7),(7,12)]:
        s = sub[(sub['wind_east'] >= lo) & (sub['wind_east'] < hi)]
        if len(s) < 10:
            continue
        print(f"    east {lo:4.1f}–{hi:4.1f} m/s: "
              f"n={len(s):4d}  util={s['Wind_Utilization'].mean():.3f}  "
              f"as_frac_of_max={s['Wind_Utilization'].mean()/s['W_east'].mean():.3f}")

# ── Correlation: per-zone vs blended ─────────────────────────────────────────
print()
print("=" * 60)
print("6. Correlation vs Wind_Utilization (west-only & east-only states)")
print("=" * 60)
for label, d in [('Pre', pre), ('Post', post)]:
    print(f"\n  [{label}]")
    for state, col in [('west_only', 'wind_west'), ('east_only', 'wind_east')]:
        s = d[d['state'] == state]
        if len(s) < 5:
            continue
        r_zone  = s[col].corr(s['Wind_Utilization'])
        r_blend = s['blend'].corr(s['Wind_Utilization'])
        print(f"    {state:12s}: corr({col})={r_zone:.3f}  corr(blend)={r_blend:.3f}  n={len(s)}")

print()
print("Done.")
