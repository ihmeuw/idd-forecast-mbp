# Project instructions — idd-forecast-mbp

## Session memory is sectioned (do not blanket-overwrite)
`.claude/memory.md` is a SHARED, SECTIONED file — sections `General` / `Malaria` /
`Dengue`, used by multiple concurrent workstreams. The session-close / `/wrap`
protocol's "overwrite memory.md" means: rewrite the file with only the section(s)
your work touched updated and **every other section copied verbatim** — never
replace the whole file with one session's snapshot.
