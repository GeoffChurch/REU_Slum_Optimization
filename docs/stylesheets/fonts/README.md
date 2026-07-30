# Self-hosted webfaces

These eight `.woff2` files back `../fonts.css`. They are committed deliberately: the site then
has **no third-party request at render time** (no `fonts.googleapis.com`, no `fonts.gstatic.com`),
which keeps visitor IPs off a CDN, removes a render-blocking external dependency from the GitHub
Pages build, and lets `mkdocs serve` render correctly offline.

| Family | Role | Files | Licence |
|---|---|---|---|
| Crimson Pro | display / headings | 2 (variable, weight axis; latin + latin-ext) | SIL Open Font License 1.1 |
| Atkinson Hyperlegible | body / UI | 6 (400, 400 italic, 700 × latin + latin-ext) | SIL Open Font License 1.1 |

Total: 184 KB. Both faces are OFL-licensed, which permits redistribution inside a project like
this one.

Atkinson Hyperlegible was commissioned by the Braille Institute and drawn to maximise
character-level distinction for low-vision readers — it is the body face specifically because
this site is dense with numeric tables where a misread digit is a misread result.

## Regenerating

Only needed when adding a weight, a style, or a non-Latin subset.

```bash
UA="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
curl -A "$UA" "https://fonts.googleapis.com/css2?family=Crimson+Pro:wght@400;600;700&family=Atkinson+Hyperlegible:ital,wght@0,400;0,700;1,400&display=swap"
```

The `User-Agent` matters: without a modern one, Google serves `.ttf` instead of `.woff2`. Take the
`latin` and `latin-ext` `@font-face` blocks from the response, download each `src` URL, and update
`../fonts.css` to match. Crimson Pro returns one variable file for every requested weight — declare
it once as `font-weight: 400 700` rather than repeating the same file per weight.
