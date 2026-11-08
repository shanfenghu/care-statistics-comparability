# Provider data and preparation

Download the public aggregate CSVs from their original providers. This repository
and its release assets do not distribute raw data or prepared copies of provider
inputs. Downloaded files, prepared data, environments and caches stay outside Git.
No account was needed for the public provider endpoints at original acquisition;
current access and version availability must be checked when running the study.
On 8 November 2026, the download command fetched all five studied versions
directly from these providers and passed verification on the tested host. This
does not guarantee future availability or access from every network.

## Sources and studied versions

Cite the Department of Health and Social Care (DHSC),
[Adult social care activity report, England: 2024 to 2025](https://www.gov.uk/government/statistics/adult-social-care-activity-report-england-2024-to-2025),
and the Department for Education (DfE),
[Children in need, reporting year 2025](https://explore-education-statistics.service.gov.uk/find-statistics/children-in-need/2025)
and [reporting year 2024](https://explore-education-statistics.service.gov.uk/find-statistics/children-in-need/2024).
The five studied downloads are:

- DHSC STS001 requests and Client Count, year ending March 2025, corrected June
  2026: [CSV](https://assets.publishing.service.gov.uk/media/6a2a75cac39255c595b506a2/requests-sts001-asc-activity-2024-to-2025-updated-june-2026.csv).
  Select ages 18–64 and 65+, pairing `Age band total` events with `Client Count`
  people. The age bands are strata of one source; people sets need not be disjoint.
- DfE B1 episodes/children, 2025 release, history 2013–2025:
  [CSV](https://explore-education-statistics.service.gov.uk/data-catalogue/data-set/37cb9a2a-7e3f-409e-8dc4-1fd399ee7d32/csv).
  Pair `Anypoint_episodes` with `Anypoint_child`.
- DfE C1 referrals/children, 2025 release, history 2013–2025:
  [CSV](https://explore-education-statistics.service.gov.uk/data-catalogue/data-set/911dae20-ac5d-4e28-bee8-53a658761fa5/csv).
  Pair `referral_count` with `referral_child_count`.
- DfE B1, 2024 release:
  [CSV](https://explore-education-statistics.service.gov.uk/data-catalogue/data-set/a33b3b51-0b9c-4774-8dc5-767a018c3ba9/csv).
- DfE C1, 2024 release:
  [CSV](https://explore-education-statistics.service.gov.uk/data-catalogue/data-set/306c2a56-228b-43f4-8493-894bee427a8c/csv).
  These two older downloads supply the preceding-release control, not independent
  sources. For 2024 C1 the columns are `Referrals` / `Referrals_child`; the
  preceding-release reader uses those names. Local 2024 values include unavailable entries.

`comparison-input-manifest.json` records filenames, original/final URLs, retrieval
times (3 November 2026), sizes and SHA-256 for the five studied versions. The DfE
catalogue versions were dated 30 October 2025 and 31 October 2024. Release landing
pages may have later updates: a release-year label alone does not identify the
CSV version. Later DHSC corrections are also different inputs. Revised child
episode definitions apply retrospectively, so earlier years in the 2025 files are
not as-of forecasts. Processor provenance for the retained adult file is unresolved.

## Download and prepare

From the repository root, after the environment setup in the root README:

```sh
.venv/bin/python experiments/comparison/acquire.py --download
.venv/bin/python experiments/comparison/acquire.py --verify
```

The first command fetches the five manifest URLs directly from DHSC/DfE, follows
provider redirects and validates the studied versions before writing any inputs
under ignored `data/raw/`. It needs about 17.1 MB of downloads. No dataset archive
from this project is required or offered. Alternatively, download the linked CSVs
manually from the providers and save them at the manifest-relative paths under
`data/raw/`. Run `--verify` afterwards. For an existing directory of provider
copies with those same relative paths, use:

```sh
.venv/bin/python experiments/comparison/acquire.py --from-dir /path/to/provider-downloads
```

If a provider endpoint is unavailable, use the original release/catalogue page to
locate that version or request it from the provider. If its bytes differ, the
command stops rather than silently analysing a revised release. Do not upload
provider data to this repository or its releases to work around access problems.
Report the unavailable version and resulting reproduction limit. A different
release needs an explicitly documented data/configuration change and scientific
comparison, rather than a claim to reproduce the original cohort.

There is no separate prepared-data package. `input_readers.py` pairs the columns
above by authority, period and family; `comparison_panel.load_rows` then keeps
local-authority rows with both families present and positive. Missing, suppressed,
unreliable and inapplicable values remain unavailable, never zero; national or
regional values do not fill local omissions. The studied current cohorts contain
152 adult and 153 child authorities. Adult nearest-five counts use radius two;
child local counts use radius zero. The temporal analysis fits 2014–2023,
calibrates on 2024 and tests 2025; fixed geographic partitions and seeds are in
`experiments/comparison/config/panel.json`. Other panels declare their own
selections in the adjacent configuration files. These steps run in the analysis:

```sh
make reproduce-comparison PYTHON=.venv/bin/python
```

Use fresh comparison outputs; the runner preserves existing results by refusing
to overwrite them. It generates results, tables and five figures locally. Plan
roughly 1 GB free disk for dependencies and outputs, and a few hundred MB of RAM;
requirements vary by platform. The tested full run took about 45 seconds on
macOS arm64 with CPython 3.9.6. Platform and actual run times are recorded locally.
The fixed-version regression comparisons use exact integer counts and floating
relative tolerance 1e-8 / absolute tolerance 1e-10; these are numerical regression
checks, not a demand for bitwise-identical archives or independent validation.

## Access, attribution and licence

Contains public sector information licensed under the
[Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).
Sources: DHSC and DfE. Crown copyright. Check the provider's current access terms,
OGL exclusions and any third-party rights; preserve attribution and do not imply
publisher endorsement. Government information is not relicensed as original MIT
software. See `THIRD-PARTY.txt` and `LICENSES/OGL-UK-3.0.txt` in the software
release. No patient-level or person-identifying records are used.
