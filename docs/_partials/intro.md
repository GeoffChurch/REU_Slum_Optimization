<!-- Handwritten Home-page partial for the multi-page site. scripts/gen_site_pages.py prepends the
     do-not-edit note, substitutes the five HTML markers below, and writes docs/index.md from
     docs/_partials/intro.md — edit HERE, never index.md (it is generated and gitignored). This
     file is committed but excluded from the built site (see exclude_docs in mkdocs.yml).

     The markers below are HTML comments named HEROLOGO, KEYRESULT, HERO, KEYFIGURES and
     METHODCOUNT. They are deliberately NOT spelled out in full here -- the generator substitutes
     by plain string replacement, so writing a marker inside this note would fill the note itself.
     HEROLOGO, KEYRESULT, HERO and KEYFIGURES are filled from run artifacts, or dropped entirely
     when those artifacts are absent, so a partial checkout never shows a placeholder. METHODCOUNT
     is not artifact-gated the same way -- it comes from the METHODS list literal in the script,
     not from disk, so it is always present:
       HEROLOGO     the official SBU mark, only if docs/brand/ actually holds it
       KEYRESULT    the headline finding, one sentence, from lens_displacement.csv
       HERO         the grown-region figure, captioned
       KEYFIGURES   four measurements from meta.json + the two lens CSVs
       METHODCOUNT  the number of published methods, as an English word (see METHODS in
                    scripts/gen_site_pages.py)

     Prose lives here; every NUMBER lives in the generator. Never type a metric into this file. -->

<section class="sbu-hero" markdown>
<div class="sbu-hero__body" markdown>

# Rebuilding access, one block at a time

<p class="sbu-hero__thesis">Roughly 1.1 billion people live in informal settlements where missing
roads cut homes off from emergency services, water, and power. <code>reblock</code> screens a whole
city for its most access-starved blocks, proposes the least-disruptive new roads to reconnect them,
and grades every proposal on the same footing.</p>

<p class="sbu-hero__result"><!-- KEYRESULT --></p>

<div class="sbu-hero__affil">
<!-- HEROLOGO -->
<p class="sbu-hero__affil-text">Stony Brook University · AI Innovation &amp; Diffusion REU</p>
</div>

</div>

<!-- HERO -->

</section>

<!-- KEYFIGURES -->

## What this measures

Every proposal is graded on one tradeoff: **permeability** — how easily every parcel can reach a
street — bought against **displacement**, the homes a road set grazes. Both are computed the same
way for every method, including the footpath network residents built themselves, so the comparison
is like-for-like rather than a scoreboard of incompatible scores.

## Start here

<div class="grid cards" markdown>

-   **[Background](background.md)**

    The problem, why roads matter, and the prior work this builds on.

-   **[Methodology](methodology.md)**

    The `data → screen → reblock` pipeline, and what the two metrics actually mean.

-   **[Methods](methods/index.md)**

    <!-- METHODCOUNT --> road-generation methods, each shown on the ground with its own numbers.

-   **[Results](benchmark.md)**

    The settlement-scale benchmark and the permeability–displacement frontier.

-   **[Team & References](team.md)**

    Who built this, and the research it stands on.

</div>

Every figure, table, and number on this site is machine-generated from run artifacts committed in
the repository — the numbers can never drift from the data.
