# Synthetic demo datasets: data card

## Registration status

The E0 request describes a user-supplied `TruthLens_synthetic_demo_datasets.zip`. The archive and its expected extracted files were not available in the repository workspace locations checked (`data`, `datasets`, `research_evidence`, and `storage`). No archive path was supplied in the request. The archive therefore cannot be registered or independently inspected in this E0.

The request records that a previous inspection found the three CSV hashes matched the archive manifest. Treat that as historical context only: this audit could not verify the archive, exact digest values, manifest contents, CSV fields, or row counts. Make the ZIP available to the local coding workspace before claiming current reproducibility or recording checksums. This card intentionally does not recreate data or infer fields.

## Expected collection (per supplied request; not locally verified)

| Expected file | Expected contents |
|---|---|
| `virality_indicators_synthetic.csv` | Synthetic virality indicators; schema unavailable until archive inspection. |
| `content_spread_trajectories_synthetic.csv` | Simulated content trajectories; schema unavailable until archive inspection. |
| `regional_spread_synthetic.csv` | Synthetic regional records; schema unavailable until archive inspection. |
| `README.md` | Expected archive documentation; contents not inspected. |
| `manifest.json` | Expected dataset manifest and checksums; contents and digest values not inspected. |
| `generate_synthetic_datasets.py` | Expected dataset generator; source and reproducibility not inspected. |

The supplied specification expects three datasets with 3,000 records in each CSV, generator version `truthlens-synthetic-demo-v1.0`, and seed `4621`. These are expected properties from the request, not verified measurements for the unavailable local archive.

## Known limitations

- The virality score is generated using a fixed heuristic formula.
- Content trajectories are simulated and grouped by content ID.
- Regional data has no generated region-specific effect.
- There is no separate human-labelled social-impact outcome dataset.

## Intended use and prohibited claims

Use these records only for clearly isolated synthetic demonstrations and interface development after the source archive and manifest are reviewed. Do not train predictive models on them. Do not present a synthetic score, trajectory, region, or social-impact signal as a real-world prediction or empirical outcome. Do not connect synthetic data to production inference.

Any later frontend demo mode must visibly show: **`SYNTHETIC DEMO DATA — NOT A REAL-WORLD PREDICTION`**.

## Checksums and verification

No checksum values are recorded because the archive and manifest were not available to this audit. Once supplied, verify the three CSV digests against `manifest.json`, count records using the generator's documented header convention, inspect the declared schema and generator version, and record the exact verified hashes here. Do not silently regenerate or repair a mismatch.
