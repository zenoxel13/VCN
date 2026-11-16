import duckdb
import pandas as pd

con = duckdb.connect("transfermarkt-datasets.duckdb", read_only=True)

query = """
WITH lineups AS (
    SELECT
        game_id,
        club_id,
        list(player_name) FILTER (WHERE type = 'starting_lineup') AS starters,
        list(player_name) FILTER (WHERE type = 'substitutes')     AS substitutes,
        list(player_name)                                         AS all_players
    FROM game_lineups
    GROUP BY game_id, club_id
)
SELECT
    g.game_id,
    g.date,
    g.competition_id,
    g.season,
    g.home_club_name,
    g.away_club_name,
    g.home_club_goals,
    g.away_club_goals,
    hl.starters    AS home_starters,
    hl.substitutes AS home_substitutes,
    hl.all_players AS home_players,
    al.starters    AS away_starters,
    al.substitutes AS away_substitutes,
    al.all_players AS away_players
FROM games g
LEFT JOIN lineups hl ON hl.game_id = g.game_id AND hl.club_id = g.home_club_id
LEFT JOIN lineups al ON al.game_id = g.game_id AND al.club_id = g.away_club_id
ORDER BY g.date
"""

df = con.sql(query).df()
print(df)

has_home = df["home_players"].notna()
has_away = df["away_players"].notna()

print(f"Total games:        {len(df)}")
print(f"Home lineup present: {has_home.sum()}")
print(f"Away lineup present: {has_away.sum()}")
print(f"Both present:        {(has_home & has_away).sum()}")
print(f"Either present:      {(has_home | has_away).sum()}")

complete = df[
    df["home_players"].notna()
    & df["away_players"].notna()
    & df["home_club_goals"].notna()
    & df["away_club_goals"].notna()
]

years = pd.to_datetime(complete["date"]).dt.year

print("First year:", years.min())
print("Last year: ", years.max())
print(df.columns.tolist())
