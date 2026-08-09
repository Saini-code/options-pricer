# CLAUDE.md

Standing rules for working in this repository.

## Code style

- Write pure functions with type hints and numpy-style docstrings.
- No notebook-style scripts in `src/` — exploratory or one-off code belongs in `notebooks/`, not in the package.
- Vectorise with numpy wherever inputs could plausibly be arrays (e.g. spot prices, strikes, maturities), rather than writing scalar-only functions and looping externally.

## Testing

- Every pricing or risk function added to `src/options_pricer/` gets a corresponding pytest unit test in `tests/`, checked against at least one known-good reference value (e.g. a textbook example or a value cross-checked independently).

## Third-party pricing libraries

- Never import a third-party options pricing library (`py_vollib`, `QuantLib`, `mibian`, or similar) into `src/`. All pricing math in `src/` must be implemented from first principles.
- Such libraries may be used in `tests/` only, as an independent cross-check against this repo's own implementations.

## Plots

- All plots are saved to `plots/` as PNG files at 150 DPI.
- Every plot must have labelled axes and a title.
