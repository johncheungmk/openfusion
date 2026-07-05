# Leaderboard Format

OpenFusion does not currently host a public leaderboard.

The result-card schema is designed so a future leaderboard can validate and compare submissions.

A responsible leaderboard should require:

- OpenFusion version;
- dataset version and hash;
- prompt template hash;
- hardware summary;
- engine and model details;
- call budget;
- strategy settings;
- accuracy or win rate;
- latency;
- call count;
- token usage.

A leaderboard should not accept unverified claims without reproducible metadata.
