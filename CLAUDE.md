# Instructions for Claude

## Git workflow: every commit lands on `main`

The team pulls `main` on their laptops, so work must not stay on a side branch.

1. Commit on the session's working branch and push it.
2. Open a pull request from that branch into `main`.
3. Merge it right away (regular merge commit). No need to ask first.
4. Before the next change, bring the working branch up to date with `origin/main`. Team members also push to `main` directly.

## Source of truth for all tasks

Follow these two documents for every design and build decision:

- `docs/Review2_Plan.md`: the Sakshya build plan (scope, architecture, models, 12-week plan)
- `docs/sakshya.html`: the Sakshya artifact (design, typefaces, stamp motif)

Where older documents (plan v2 PDF, Review 1 deck) disagree, these two win. Keep claims limited to what is evaluated, and treat the artifact's example numbers as illustrative.

## App layout

- `backend/`: FastAPI. Run the tests with `cd backend && .venv/bin/python -m pytest`.
- `frontend/`: Next.js + TypeScript. Check it with `npm run build`.
- Run instructions are in `README.md`.
