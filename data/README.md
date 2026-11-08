# Exact event/person inputs

Five publisher CSV versions are required. `comparison-input-manifest.json`
records exact byte sizes, SHA-256, retrieval times, publisher terms and original
and final URLs. The total is 17,071,519 bytes. Raw inputs stay outside Git.

- DHSC adult social care activity, reporting year 2024–2025: STS001 requests and
  Client Count, the retained June 2026 corrected CSV. Select ages 18–64 and 65+,
  whole-age-band event totals and their Client Count counterparts. They are two
  strata of one source, and people sets are not asserted disjoint. Later October
  corrections are a different version. Four-day/seven-day clustering descriptions
  do not establish which processor produced this retained file.
- DfE B1 episodes/children and C1 referrals/children, both published/updated
  30 October 2025, with history 2013–2025. These are distinct targets within one
  child source. Local-authority values are unrounded; revised episode definitions
  apply retrospectively. Earlier-year tests on these files are not as-of forecasts.
- DfE B1/C1 reporting-year-2024 versions, both published/updated 31 October 2024.
  They supply the separate preceding-release control, not extra independent sources.

The current complete positive cohorts contain 152 adult and 153 child authorities.
Both families must be present and positive. Missing, suppressed, unreliable and
inapplicable values remain unavailable. Local omissions cannot be filled with
national/regional substitutions. Adult nearest-five publication uses radius two;
child local counts use radius zero. Passing count coherence does not validate
administrative returns or processing.

From the repository root, use one acquisition mode:

```sh
.venv/bin/python experiments/comparison/acquire.py --archive /path/to/aggregate-input-snapshot-1.0.2.tar.gz
.venv/bin/python experiments/comparison/acquire.py --from-dir /path/to/retained-inputs
.venv/bin/python experiments/comparison/acquire.py --download
.venv/bin/python experiments/comparison/acquire.py --verify
```

The first two commands are offline alternatives to the third.
The HTTP mode is optional and was not exercised for this version; current
publisher availability is unverified. The archive is
941,808 bytes, SHA-256 `8e0c3a5f3ce5829089fe5a4452bb86bed34c5b711edc4840526e18886ae137ce`.
It preserves the five CSVs unchanged plus four attribution/licence/guide/manifest
files. Acquisition writes only verified CSVs at the manifest-relative paths.
It checks all sources and existing destinations before changing any input.
Version 1.0.2 updates two wrapper files relative to 1.0.1; all five CSVs,
attribution and licence remain unchanged from both earlier versions.

If publisher URLs fail or change, obtain the exact retained snapshot through its
versioned release asset or a lawful retained copy. A URL/checksum is not a backup;
this software does not promise that changing publisher endpoints retain old bytes.
The archive must accompany a complete distribution. Local reproducibility alone
does not establish public archive access or independent preservation.

Contains public sector information licensed under the Open Government Licence
v3.0. Sources: Department of Health and Social Care and Department for Education.
Crown copyright. Retain source attribution, OGL exclusions and applicable
third-party rights; no publisher endorsement is implied. See `THIRD-PARTY.txt`
and `LICENSES/OGL-UK-3.0.txt`. No patient-level or person-identifying records are used.
