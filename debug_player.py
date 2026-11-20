#!/usr/bin/env python3
"""
Prints a full breakdown of the projection for one or more players.

Usage:  python debug_player.py Haaland Gabriel
"""
import sys
import fetch_data as m


def main(names):
    bootstrap = m.get_json(f"{m.BASE}/bootstrap-static/")
    fixtures = m.get_json(f"{m.BASE}/fixtures/")
    teams = bootstrap["teams"]
    team_by_id = {t["id"]: t for t in teams}
    attack, defence, avg_att, avg_def = m.build_team_stats(teams, bootstrap["elements"], fixtures)

    priors = m.bonus_priors(bootstrap["elements"])
    next_event = next(e["id"] for e in bootstrap["events"] if e.get("is_next"))
    print(f"Next GW: {next_event}")
    print(f"League avg xG for/game: {avg_att:.2f}   league avg xGC/game: {avg_def:.2f}\n")

    for name in names:
        matches = [p for p in bootstrap["elements"]
                   if name.lower() in p["web_name"].lower() and p["minutes"] > 0]
        for p in matches:
            hist = m.get_json(f"{m.BASE}/element-summary/{p['id']}/")["history"]
            prof = m.build_profile(p, hist, priors[m.POSITIONS[p["element_type"]]])
            tid = p["team"]
            t = team_by_id[tid]
            pos = prof["pos"]
            mins = p["minutes"]
            recent = hist[-m.FORM_GAMES:]
            rmin = sum(h["minutes"] for h in recent)

            print("=" * 70)
            print(f"{p['web_name']} ({t['short_name']}, {pos})  |  FPL points/game this season: "
                  f"{p['total_points'] / max(1, len([h for h in hist if h['minutes'] > 0])):.2f}"
                  f"  |  bonus/90: {p['bonus'] * 90 / mins:.2f}")
            print(f"Season: {mins} min | xG {p['expected_goals']} | G {p['goals_scored']} | "
                  f"xA {p['expected_assists']} | A {p['assists']}")
            print(f"Last {len(recent)} rows: {rmin} min | xG {sum(m.f(h['expected_goals']) for h in recent):.2f} | "
                  f"G {sum(h['goals_scored'] for h in recent)} | "
                  f"avg min {rmin / max(1, len(recent)):.0f} -> fringe={prof['fringe']}")
            print(f"Blended per-90: goals {prof['g90']:.3f} | assists {prof['a90']:.3f} | "
                  f"defcon hit rate {prof['dc']:.2f} | saves {prof['saves90']:.2f}")
            print(f"Team: xG for/game {attack[tid]:.2f} | xGC/game {defence[tid]:.2f}")

            my_fx = sorted(
                (fx for fx in fixtures
                 if fx.get("event") and next_event <= fx["event"] < next_event + m.HORIZON
                 and tid in (fx["team_h"], fx["team_a"])),
                key=lambda fx: fx["event"])
            running = 0.0
            for idx, fx in enumerate(my_fx):
                home = fx["team_h"] == tid
                opp = fx["team_a"] if home else fx["team_h"]
                points, info = m.fixture_components(prof, tid, opp, home, attack, defence, avg_att, avg_def)
                total = sum(points.values())
                running += total
                label = f"GW{fx['event']} vs {team_by_id[opp]['short_name']} ({'H' if home else 'A'})"
                if idx == 0:
                    print(f"\nNext fixture: {label}")
                    if info:
                        print(f"  opp xGC/game {defence[opp]:.2f} -> attack mult {info['att_mult']:.3f} | "
                              f"opp xG/game {attack[opp]:.2f} -> defence mult {info['def_mult']:.3f}")
                        print(f"  lambda conceded {info['lambda_conceded']:.2f} -> "
                              f"P(clean sheet) {info['p_clean_sheet']:.1%}")
                    for k, v in points.items():
                        print(f"  {k:<16}{v:6.2f}")
                    print(f"  {'TOTAL':<16}{total:6.2f}")
                    print("\nAll fixtures in horizon:")
                if info:
                    print(f"  {label:<22} total {total:5.2f} | att mult {info['att_mult']:.2f} | "
                          f"CS {info['p_clean_sheet']:.0%} | G+A {info['p_involve']:.0%}")
                else:
                    print(f"  {label:<22} total {total:5.2f} | fringe default")
            print(f"  Sum over {len(my_fx)} fixtures: {running:.2f}")
            print()


if __name__ == "__main__":
    main(sys.argv[1:] or ["Haaland", "Gabriel"])