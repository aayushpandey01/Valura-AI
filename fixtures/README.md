# Fixtures

This directory contains all test data for the Valura AI microservice.

## Structure

```
fixtures/
  user_profiles/          — 5 user profiles for testing
  conversations/          — 3 conversation transcripts
  test_queries/
    intent_classification.json   — ~60 labeled classification queries
    safety_pairs.json            — ~45 safety queries (harmful + educational)
```

## User Profiles

| ID | Profile |
|----|---------|
| user_001 | Aggressive trader, concentrated tech |
| user_002 | Concentrated single-stock holder (NVDA) |
| user_003 | Multi-currency global investor |
| user_004 | Empty portfolio (new user, no holdings) |
| user_005 | Dividend-focused retiree |

## Entity Normalization Rules

- **Tickers**: case-fold before comparison (`AAPL` == `aapl`). Exchange suffix optional: `ASML` matches `ASML.AS`
- **Amounts**: numeric match within ±5%
- **Periods**: `period_years` match within ±5%
- **String lists**: subset match — your output must include every expected value; extras allowed
- **Topics/sectors**: lowercase comparison
