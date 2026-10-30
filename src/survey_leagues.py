import collections

import requests

B = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
comps = requests.get(B + "/competitions.json", timeout=60).json()
want = {"1. Bundesliga", "Premier League", "Ligue 1", "La Liga", "Serie A"}
for c in comps:
    if c["competition_name"] in want and c["competition_gender"] == "male":
        m = requests.get(f"{B}/matches/{c['competition_id']}/{c['season_id']}.json", timeout=60).json()
        cnt = collections.Counter()
        for x in m:
            cnt[x["home_team"]["home_team_name"]] += 1
            cnt[x["away_team"]["away_team_name"]] += 1
        top, n = cnt.most_common(1)[0]
        print(c["competition_name"], c["season_name"], c["competition_id"], c["season_id"],
              len(m), "matches;", len(cnt), "teams; top", top, n, "; 360:", c["match_available_360"] is not None)
