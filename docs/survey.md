# survey

Scan many sectors, cameras and CCDs quickly. Every star gets a cheap look, and only the few that show
something get the full `hunt`.

```bash
survey plan --dir surveys/s70-75 --sectors 70-75            # 96 CCD jobs (add --cameras 1,2 --ccds 3 to narrow)
survey run  --dir surveys/s70-75 --detach                   # background; or without --detach in 150 s chunks (exit 75 = paused)
survey status --dir surveys/s70-75                          # one line per CCD
survey top  --dir surveys/s70-75 -n 20                      # best leads across the survey
survey summary --dir surveys/s70-75                         # prints survey_summary.md
survey reclassify --dir surveys/s70-75                      # re-apply class rules (cached catalog checks)
```

`plan` can be re-run with more sectors: jobs are only ever added, and finished CCDs stay finished.

## The three tiers

| tier | who | what | cost |
|---|---|---|---|
| quick | every selected star on every CCD | That sector's TESS-SPOC light curve, from a URL built from the TIC (no MAST search). Plus `--quick-sectors − 1` earlier sectors from tess-point. Detrend, banded BLS (pmin 0.5 d), single-dip search. The file is deleted unless the best signal has SNR ≥ 6. | ~1.2 MB and ~0.4 s CPU per star; downloads (`--dl-threads` 6) overlap with searches (`--workers`) |
| flag | per CCD | Bad windows (many stars dipping together) are found from all stars, and signals in them are dropped. Flags a periodic signal (SNR ≥ 7, ≥ 2 transits, not a harmonic or integer day) or a clean single dip (SNR ≥ 8). The top `--max-deep` (150) go deep. | seconds |
| deep | flagged stars | `hunt` with every sector: duo funnel, full BLS, triage, gate, vet (packets with centroid). It gets the CCD's bad windows. | minutes per CCD |

Then **classify** sorts the leads and checks the strongest against the catalogs: ExoFOP TOI/CTOI, NASA,
Prsa+2022 EBs, and Gaia DR3 EBs and non-single stars.

| class | meaning |
|---|---|
| planet_candidate | periodic signal that passed gate, catalog, secondary and centroid |
| duo_candidate | two single transits years apart with aliases solved, not explained by a periodic signal |
| single_transit | one clean deep dip and no repeat in the data: a long-period planet, brown dwarf, EB, or something odder |
| eb_candidate | strong periodic signal (SNR ≥ 12) with eclipse evidence: a secondary eclipse, a grazing/V-shaped T14 (too short for the star), or odd/even or size failures on a signal that is neither sinusoid-like nor too long |

Spotted, rotating and pulsating stars also fail the planet tests, with wide sinusoid-like boxes and T14
far too long for the star. They're counted in the `variables` column and kept off the leaderboard. On
the first live slice (S74/1/2, 60 brightest stars) a looser rule called 8 stars EBs; 7 were rotators.
The rule now keeps only TIC 99725941 (the Gaia SB2).

`survey reclassify --dir D` re-applies the classify rules to finished CCDs. It uses the cached catalog
checks, so it takes seconds.

**novelty** values:
- `new`: no catalog lists the star as a planet host, TOI/CTOI or EB.
- `new_eclipses_known_binary`: Gaia NSS knows it's a binary, but no catalog has the eclipses.
- `new (EB nearby)`, `known_*`, `unchecked`.

`--max-veto` (15) EB and single leads per CCD are checked; the rest say `unchecked (below --max-veto)`.

## What it can't see

The quick tier only sees signals strong in `--quick-sectors` sectors (default 1): deep eclipses, giant
planets, strong single dips. Shallow signals that need many sectors are missed. TIC 141618383's 10.43 d
signal is SNR 13 over 22 transits but about 5 in S74 alone. Raising `--quick-sectors` to 3 recovers a lot
of that at about 3× the download. For the deepest look at one CCD, use `hunt`.

## Throughput (measured 22 Nov 2026, this box, 2 CPUs)

The quick tier ran at about 1.4 stars/s on S74/1/2, download-bound through the MAST API fallback (the
archive.stsci.edu mirror was down). That's about 30 min per 2,500-star CCD, or about 8 h per sector of
16 CCDs. With the mirror up and more threads it's faster; MAST throttles at about 12.

## Robustness

- A CCD that fails is retried on the next `survey run`, up to `--max-attempts` (3). Cached errors are cleared first.
- When archive.stsci.edu resets connections, `tess_lc_fetch` skips it for 10 minutes and uses the MAST download API. Target lists fall back the same way.
- A failed control in the deep tier stops the survey (exit 2).

## Outputs (`<dir>/`)

- `leaderboard.csv`: every lead, best first. Order: class, then new before known, then SNR.
- `survey_summary.md`: one table row per CCD, then the top 25 leads.
- `jobs/<sNN_cC_dD>/`: `flagged.json` (bad windows, flags), `stages/quick/`, `deep/` (a full hunt run dir with its own `summary.md`), `classify.json`.

"new" means "not in these catalogs today". It doesn't mean discovered. Every lead needs a human review and
follow-up before anyone calls it a planet.
